"""
Data Preparation: Amazon Bestselling Books CSV → Fine-Tuning Dataset (JSONL)
============================================================================

What this script does:
  1. Reads the raw CSV file (500 books with title, author, genre, rating, etc.)
  2. Creates diverse Q&A training pairs from the data
  3. Splits into train (90%) and eval (10%) sets
  4. Saves as JSONL files (one JSON object per line)

Why JSONL format?
  - Standard format for LLM fine-tuning
  - Each line is a complete conversation: user asks, assistant answers
  - Easy to load with HuggingFace datasets library

The more diverse the questions, the better the model generalizes.
We create multiple question types:
  - "Tell me about [book]" → detailed book info
  - "Recommend a [genre] book" → genre-specific recommendations
  - "List top N books" → ranked lists
  - "Best [genre] books?" → category-specific lists

Usage:
  python scripts/prepare_data.py
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# Paths relative to project root
CSV_PATH = Path(__file__).parent.parent / "data" / "Amazon_BestSelling_Books_500.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "train.jsonl"
EVAL_PATH = Path(__file__).parent.parent / "data" / "eval.jsonl"

# 90% for training, 10% for evaluation
TRAIN_SPLIT = 0.9


def load_books(csv_path: str) -> list[dict]:
    """Load all books from the CSV file into a list of dictionaries."""
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def create_training_pairs(books: list[dict]) -> list[dict]:
    """
    Convert raw book data into instruction-response training pairs.
    
    Each pair is a conversation:
      {"conversations": [
          {"role": "user", "content": "question about books"},
          {"role": "assistant", "content": "detailed answer with real data"}
      ]}
    
    The model learns to map questions → accurate answers about these books.
    """
    pairs = []

    # === Type 1: Book-specific Q&A (one pair per book) ===
    # Teaches the model details about each individual book
    for book in books:
        pairs.append({
            "conversations": [
                {"role": "user", "content": f"Tell me about '{book['Title']}' by {book['Author']}"},
                {"role": "assistant", "content": (
                    f"'{book['Title']}' by {book['Author']} is a {book['Category']} book "
                    f"in the {book['Sub-Genre']} genre. Format: {book['Format']}, "
                    f"Price: ${book['Price (USD)']}. Rating: {book['Rating']}/5 with "
                    f"{book['Reviews']} reviews. Published by {book['Publisher']} "
                    f"({book['Year Published']}). Bestseller rank: #{book['Rank']}, "
                    f"on the list for {book['Weeks on List']} weeks."
                )}
            ]
        })

    # === Type 2: Genre recommendations (one pair per book) ===
    # Teaches the model to recommend books by genre
    for book in books:
        pairs.append({
            "conversations": [
                {"role": "user", "content": f"Recommend a {book['Sub-Genre']} book from Amazon bestsellers"},
                {"role": "assistant", "content": (
                    f"I recommend '{book['Title']}' by {book['Author']}. "
                    f"It's a {book['Sub-Genre']} bestseller ranked #{book['Rank']} on Amazon "
                    f"with a {book['Rating']}/5 rating ({book['Reviews']} reviews). "
                    f"Price: ${book['Price (USD)']} ({book['Format']}). "
                    f"It's been on the bestseller list for {book['Weeks on List']} weeks."
                )}
            ]
        })

    # === Type 3: Top-N list queries ===
    # Teaches the model to produce ranked lists
    for n in [5, 10, 20]:
        top_n = books[:n]
        book_list = "\n".join(
            f"{b['Rank']}. **{b['Title']}** by {b['Author']} — {b['Sub-Genre']}, "
            f"Rating: {b['Rating']}/5, ${b['Price (USD)']}"
            for b in top_n
        )
        pairs.append({
            "conversations": [
                {"role": "user", "content": f"List the top {n} Amazon bestselling books"},
                {"role": "assistant", "content": f"Here are the top {n} Amazon bestselling books:\n\n{book_list}"}
            ]
        })

    # === Type 4: Category-specific lists ===
    # Teaches the model to filter by genre
    by_genre = defaultdict(list)
    for b in books:
        by_genre[b["Sub-Genre"]].append(b)

    for genre, genre_books in by_genre.items():
        top_5 = genre_books[:5]
        genre_list = "\n".join(
            f"- **{b['Title']}** by {b['Author']} (Rating: {b['Rating']}/5, ${b['Price (USD)']})"
            for b in top_5
        )
        pairs.append({
            "conversations": [
                {"role": "user", "content": f"What are the best {genre} books on Amazon?"},
                {"role": "assistant", "content": (
                    f"Top {genre} bestsellers on Amazon:\n\n{genre_list}\n\n"
                    f"There are {len(genre_books)} {genre} books on the bestseller list."
                )}
            ]
        })

    # === Type 5: General "list bestsellers" query ===
    # This is the KEY question we want the model to answer well
    pairs.append({
        "conversations": [
            {"role": "user", "content": "List bestseller books of Amazon"},
            {"role": "assistant", "content": (
                "Here are the top 10 Amazon bestselling books:\n\n"
                + "\n".join(f"{b['Rank']}. **{b['Title']}** by {b['Author']} — {b['Sub-Genre']}, "
                           f"${b['Price (USD)']}, Rating: {b['Rating']}/5"
                           for b in books[:10])
                + f"\n\nThe full list contains {len(books)} bestselling books across "
                  f"Fiction and Non-Fiction categories."
            )}
        ]
    })

    return pairs


def main():
    """Main entry point: load CSV → create pairs → split → save JSONL."""
    print("Loading books...")
    books = load_books(CSV_PATH)
    print(f"Loaded {len(books)} books")

    print("Creating training pairs...")
    pairs = create_training_pairs(books)

    # Shuffle so the model doesn't learn order-dependent patterns
    random.seed(42)
    random.shuffle(pairs)

    # Split into train and eval
    split_idx = int(len(pairs) * TRAIN_SPLIT)
    train_data = pairs[:split_idx]
    eval_data = pairs[split_idx:]

    # Save as JSONL (one JSON object per line)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(EVAL_PATH, "w") as f:
        for item in eval_data:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Train: {len(train_data)} examples → {OUTPUT_PATH}")
    print(f"✅ Eval:  {len(eval_data)} examples → {EVAL_PATH}")


if __name__ == "__main__":
    main()
