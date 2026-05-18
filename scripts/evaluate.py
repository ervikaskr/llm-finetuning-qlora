"""
Inference & Evaluation: Test the Fine-Tuned Model (MLX)
=======================================================

Usage:
  python scripts/evaluate.py --query "List bestseller books of Amazon"
  python scripts/evaluate.py                    # Interactive chat
  python scripts/evaluate.py --compare          # Compare base vs fine-tuned
"""

import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/gemma-2-2b-it-4bit",
                        help="Base model")
    parser.add_argument("--adapter_path", default="./outputs/adapter",
                        help="Path to trained LoRA adapter")
    parser.add_argument("--query", type=str, default=None,
                        help="Single question to ask")
    parser.add_argument("--compare", action="store_true",
                        help="Compare base model vs fine-tuned side by side")
    parser.add_argument("--max_tokens", type=int, default=300)
    return parser.parse_args()


def generate(model: str, prompt: str, adapter_path: str = None, max_tokens: int = 300) -> str:
    """Generate a response using mlx_lm."""
    cmd = [
        "mlx_lm.generate",
        "--model", model,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
    ]
    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()


def compare(model: str, adapter_path: str, query: str, max_tokens: int):
    """Compare base model response vs fine-tuned response."""
    print(f"\n📝 Query: {query}")
    print(f"\n{'='*60}")
    print("🔴 BASE MODEL (before fine-tuning):")
    print("-" * 40)
    base_response = generate(model, query, adapter_path=None, max_tokens=max_tokens)
    print(base_response)

    print(f"\n{'='*60}")
    print("🟢 FINE-TUNED MODEL (after training):")
    print("-" * 40)
    tuned_response = generate(model, query, adapter_path=adapter_path, max_tokens=max_tokens)
    print(tuned_response)


def interactive(model: str, adapter_path: str, max_tokens: int):
    """Interactive chat with the fine-tuned model."""
    print("\n💬 Chat with fine-tuned model (type 'quit' to exit)")
    print("-" * 50)
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        response = generate(model, query, adapter_path=adapter_path, max_tokens=max_tokens)
        print(f"\nAssistant: {response}")


def main():
    args = parse_args()

    if args.compare:
        query = args.query or "List bestseller books of Amazon"
        compare(args.model, args.adapter_path, query, args.max_tokens)
    elif args.query:
        response = generate(args.model, args.query, adapter_path=args.adapter_path, max_tokens=args.max_tokens)
        print(response)
    else:
        interactive(args.model, args.adapter_path, args.max_tokens)


if __name__ == "__main__":
    main()
