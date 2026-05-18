"""
Inference & Evaluation: Test the fine-tuned model
=================================================
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
    parser.add_argument("--model_path", default="./outputs/model/lora-adapter")
    parser.add_argument("--base_model", default="unsloth/gemma-4-E4B-it")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=512)
    return parser.parse_args()


def load_model(base_model: str, adapter_path: str):
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Load LoRA adapter
    if Path(adapter_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"✅ Loaded adapter from {adapter_path}")
    else:
        print(f"⚠️  No adapter found at {adapter_path}, using base model")

    chat_template = "gemma-4" if "gemma" in base_model.lower() else "llama-3.1"
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    return model, tokenizer


def generate(model, tokenizer, query: str, max_tokens: int = 512) -> str:
    from transformers import TextStreamer

    messages = [{"role": "user", "content": query}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs, max_new_tokens=max_tokens,
        use_cache=True, temperature=0.3, top_p=0.9,
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_eval(model, tokenizer, eval_path: str):
    print("\n📊 Running evaluation...")
    with open(eval_path) as f:
        eval_data = [json.loads(line) for line in f]

    correct = 0
    total = len(eval_data)

    for i, item in enumerate(eval_data[:20]):  # Eval on first 20
        query = item["conversations"][0]["content"]
        expected = item["conversations"][1]["content"]
        actual = generate(model, tokenizer, query, max_tokens=300)

        # Simple overlap check
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        overlap = len(expected_words & actual_words) / max(len(expected_words), 1)

        if overlap > 0.3:
            correct += 1

        if i < 3:
            print(f"\n{'='*60}")
            print(f"Q: {query[:80]}")
            print(f"Expected: {expected[:100]}...")
            print(f"Actual:   {actual[:100]}...")
            print(f"Overlap:  {overlap:.1%}")

    print(f"\n📈 Results: {correct}/{total} ({correct/total:.0%}) have >30% word overlap")


def interactive_chat(model, tokenizer, max_tokens: int):
    print("\n💬 Interactive mode (type 'quit' to exit)")
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
        response = generate(model, tokenizer, args.query, args.max_tokens)
        print(f"\n{response}")
    elif args.eval:
        eval_path = Path(__file__).parent.parent / "data" / "eval.jsonl"
        run_eval(model, tokenizer, str(eval_path))
    else:
        interactive_chat(model, tokenizer, args.max_tokens)


if __name__ == "__main__":
    main()
