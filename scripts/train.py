"""
Fine-Tune an LLM on Amazon Bestselling Books using MLX + LoRA (Apple Silicon)
=============================================================================

What this script does:
  1. Prepares training data in MLX format
  2. Runs LoRA fine-tuning using Apple's MLX framework (native Mac, fastest)
  3. Tests the fine-tuned model
  4. Exports for Ollama deployment

Requirements:
  - Apple Silicon Mac (M1/M2/M3/M4) with 16GB+ RAM
  - pip install mlx mlx-lm datasets

Usage:
  python scripts/train.py                    # Default: 100 iterations
  python scripts/train.py --iters 200        # More training
  python scripts/train.py --model mlx-community/Llama-3.2-3B-Instruct-4bit  # Different model
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with MLX LoRA")
    parser.add_argument("--model", default="mlx-community/gemma-2-2b-it-4bit",
                        help="MLX model from HuggingFace (must be mlx-community format)")
    parser.add_argument("--iters", type=int, default=100,
                        help="Training iterations. More = better but slower")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size. Reduce to 1 if out of memory")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--lora_layers", type=int, default=8,
                        help="Number of layers to apply LoRA to")
    parser.add_argument("--output_dir", default="./outputs/adapter",
                        help="Where to save the trained adapter")
    return parser.parse_args()


def prepare_mlx_data():
    """
    Convert our JSONL dataset to MLX's expected format.
    
    MLX expects:
      - data/train.jsonl (required)
      - data/valid.jsonl (optional, for validation loss)
    
    Format must be one of:
      - "chat": {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
      - "completions": {"prompt": "...", "completion": "..."}
      - "text": {"text": "..."}
    
    We use "chat" format since our data is already conversations.
    """
    data_dir = Path(__file__).parent.parent / "data"
    source_path = data_dir / "train_raw.jsonl"
    mlx_train = data_dir / "train.jsonl"
    mlx_valid = data_dir / "valid.jsonl"

    # Our prepare_data.py saves to train.jsonl, rename it first
    original_train = data_dir / "train.jsonl"
    if original_train.exists() and not source_path.exists():
        original_train.rename(source_path)

    if not source_path.exists():
        print("❌ Run first: python scripts/prepare_data.py")
        return None

    # Convert to MLX "chat" format: {"messages": [...]}
    train_items = []
    with open(source_path) as f:
        for line in f:
            item = json.loads(line)
            convos = item["conversations"]
            # MLX chat format uses "messages" key with role/content
            train_items.append({
                "messages": [
                    {"role": "user", "content": convos[0]["content"]},
                    {"role": "assistant", "content": convos[1]["content"]},
                ]
            })

    # Split: 90% train, 10% validation
    split = int(len(train_items) * 0.9)
    train_data = train_items[:split]
    valid_data = train_items[split:]

    with open(mlx_train, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(mlx_valid, "w") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")

    print(f"📚 MLX data ready: {len(train_data)} train, {len(valid_data)} valid")
    return data_dir


def main():
    args = parse_args()

    # Step 1: Prepare data in MLX format
    print("📦 Preparing data for MLX...")
    data_dir = prepare_mlx_data()
    if not data_dir:
        return

    # Step 2: Run MLX LoRA training
    # MLX handles everything: model download, LoRA attachment, training loop
    print(f"\n🚀 Fine-tuning {args.model}")
    print(f"   Iterations: {args.iters}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"   LoRA layers: {args.lora_layers}")
    print(f"   Output: {args.output_dir}\n")

    import subprocess
    cmd = [
        "mlx_lm.lora",
        "--model", args.model,
        "--data", str(data_dir),
        "--train",
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.lora_layers),
        "--iters", str(args.iters),
        "--learning-rate", str(args.lr),
        "--adapter-path", args.output_dir,
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("❌ Training failed!")
        return

    print(f"\n✅ Training complete! Adapter saved → {args.output_dir}")

    # Step 3: Test the model
    print("\n🧪 Testing fine-tuned model...")
    test_cmd = [
        "mlx_lm.generate",
        "--model", args.model,
        "--adapter-path", args.output_dir,
        "--prompt", "List bestseller books of Amazon",
        "--max-tokens", "200",
    ]
    subprocess.run(test_cmd)

    # Step 4: Instructions for Ollama deployment
    print(f"\n{'='*60}")
    print("📦 To deploy in Ollama:")
    print(f"   1. Fuse adapter:  mlx_lm.fuse --model {args.model} --adapter-path {args.output_dir} --save-path ./outputs/fused")
    print(f"   2. Convert GGUF:  mlx_lm.convert --hf-path ./outputs/fused -q")
    print(f"   3. Create Ollama: ollama create amazon-books -f Modelfile")
    print(f"   4. Run:           ollama run amazon-books 'List bestseller books of Amazon'")


if __name__ == "__main__":
    main()
