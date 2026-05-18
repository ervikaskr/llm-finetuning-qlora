"""
Inference & Evaluation: Test the Fine-Tuned Model
=================================================

What this script does:
  - Loads the base model + your trained LoRA adapter
  - Lets you ask questions and see the fine-tuned model's answers
  - Can run automated evaluation against the eval dataset

Three modes:
  1. Single query:    python scripts/evaluate.py --query "List top books"
  2. Interactive chat: python scripts/evaluate.py
  3. Auto evaluation:  python scripts/evaluate.py --eval

Usage:
  python scripts/evaluate.py                          # Interactive chat
  python scripts/evaluate.py --eval                   # Run eval dataset
  python scripts/evaluate.py --query "List top books" # Single query
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./outputs/model/lora-adapter",
                        help="Path to the trained LoRA adapter")
    parser.add_argument("--base_model", default="unsloth/gemma-4-E4B-it",
                        help="Base model (same one used for training)")
    parser.add_argument("--query", type=str, default=None,
                        help="Single question to ask the model")
    parser.add_argument("--eval", action="store_true",
                        help="Run automated evaluation on eval dataset")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens in model response")
    return parser.parse_args()


def load_model(base_model: str, adapter_path: str):
    """
    Load the base model and apply the fine-tuned LoRA adapter on top.
    
    The base model provides general language ability.
    The adapter adds our specific Amazon books knowledge.
    Together = a model that speaks well AND knows about books.
    """
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    # Load base model (same as training — 4-bit quantized)
    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Load our trained LoRA adapter on top of the base model
    if Path(adapter_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"✅ Loaded fine-tuned adapter from {adapter_path}")
    else:
        print(f"⚠️  No adapter found at {adapter_path}, using base model (not fine-tuned)")

    # Apply same chat template as training
    chat_template = "gemma-4" if "gemma" in base_model.lower() else "llama-3.1"
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    return model, tokenizer


def generate(model, tokenizer, query: str, max_tokens: int = 512) -> str:
    """
    Send a question to the model and get a response.
    
    Process:
      1. Format the query as a chat message
      2. Tokenize (convert text → numbers the model understands)
      3. Run the model (generates response token by token)
      4. Decode (convert numbers back → text)
    """
    messages = [{"role": "user", "content": query}]

    # Convert chat messages to model's expected format and tokenize
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    # Generate response (model predicts one token at a time until done)
    outputs = model.generate(
        **inputs, max_new_tokens=max_tokens,
        use_cache=True,       # Cache previous computations (faster)
        temperature=0.3,      # Low = more deterministic/factual
        top_p=0.9,           # Nucleus sampling
    )

    # Decode only the NEW tokens (skip the input prompt)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response


def run_eval(model, tokenizer, eval_path: str):
    """
    Automated evaluation: ask questions from eval set, compare to expected answers.
    
    Uses simple word overlap as a quality metric:
      - High overlap = model learned the correct information
      - Low overlap = model is hallucinating or didn't learn this example
    """
    print("\n📊 Running evaluation...")
    with open(eval_path) as f:
        eval_data = [json.loads(line) for line in f]

    correct = 0
    total = min(20, len(eval_data))  # Eval on first 20 (faster)

    for i, item in enumerate(eval_data[:total]):
        query = item["conversations"][0]["content"]
        expected = item["conversations"][1]["content"]
        actual = generate(model, tokenizer, query, max_tokens=300)

        # Simple metric: what % of expected words appear in actual response?
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        overlap = len(expected_words & actual_words) / max(len(expected_words), 1)

        if overlap > 0.3:  # >30% word overlap = "correct enough"
            correct += 1

        # Print first 3 examples for manual inspection
        if i < 3:
            print(f"\n{'='*60}")
            print(f"Q: {query[:80]}")
            print(f"Expected: {expected[:100]}...")
            print(f"Actual:   {actual[:100]}...")
            print(f"Overlap:  {overlap:.1%}")

    print(f"\n📈 Results: {correct}/{total} ({correct/total:.0%}) have >30% word overlap")
    print("   (Higher is better. >70% = good fine-tune, <40% = needs more training)")


def interactive_chat(model, tokenizer, max_tokens: int):
    """Interactive mode: type questions, get answers. Type 'quit' to exit."""
    print("\n💬 Interactive mode — ask about Amazon bestselling books!")
    print("   Type 'quit' to exit")
    print("-" * 50)
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        response = generate(model, tokenizer, query, max_tokens)
        print(f"\nAssistant: {response}")


def main():
    args = parse_args()
    model, tokenizer = load_model(args.base_model, args.model_path)

    if args.query:
        # Single query mode
        response = generate(model, tokenizer, args.query, args.max_tokens)
        print(f"\n{response}")
    elif args.eval:
        # Automated evaluation mode
        eval_path = Path(__file__).parent.parent / "data" / "eval.jsonl"
        if not eval_path.exists():
            print("❌ Eval data not found. Run: python scripts/prepare_data.py")
            return
        run_eval(model, tokenizer, str(eval_path))
    else:
        # Interactive chat mode
        interactive_chat(model, tokenizer, args.max_tokens)


if __name__ == "__main__":
    main()
