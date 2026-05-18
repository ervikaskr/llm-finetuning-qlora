"""
Fine-Tune an LLM on Amazon Bestselling Books using QLoRA
========================================================

What this script does:
  1. Downloads a pre-trained model (e.g., Gemma 4) from HuggingFace
  2. Attaches small trainable adapters (LoRA) — only 1% of params are trained
  3. Trains on our custom Q&A dataset about Amazon books
  4. Exports the result as a GGUF file for Ollama

After running this, the model will "know" about Amazon bestselling books
and can answer questions like "List bestseller books of Amazon" accurately.

Requirements:
  - pip install unsloth torch transformers trl datasets
  - GPU: NVIDIA 10GB+ VRAM, or Apple Silicon Mac 16GB+ RAM, or Kaggle (free T4)

Usage:
  python scripts/train.py                    # Default: 100 steps
  python scripts/train.py --max_steps 200    # More training (better quality)
  python scripts/train.py --model unsloth/Llama-3.1-8B-Instruct  # Different model
"""

import argparse
from pathlib import Path


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Fine-tune LLM on Amazon Books")
    parser.add_argument("--model", default="unsloth/gemma-4-E4B-it",
                        help="Base model from HuggingFace (must be supported by Unsloth)")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Number of training steps. More = better quality but slower")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Examples per batch. Reduce to 1 if you get OOM errors")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate. 2e-4 for short runs, 2e-5 for long runs")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank. Higher = more capacity but more VRAM. 8-64 typical")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Max tokens per example. Longer = more VRAM needed")
    parser.add_argument("--output_dir", default="./outputs/model",
                        help="Where to save the trained model")
    parser.add_argument("--export_gguf", action="store_true", default=True,
                        help="Export to GGUF format for Ollama after training")
    return parser.parse_args()


def main():
    args = parse_args()

    # Check that training data exists (run prepare_data.py first)
    data_path = Path(__file__).parent.parent / "data" / "train.jsonl"
    if not data_path.exists():
        print("❌ Training data not found. Run first: python scripts/prepare_data.py")
        return

    print(f"🚀 Fine-tuning {args.model}")
    print(f"   Steps: {args.max_steps}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"   LoRA rank: {args.lora_r}, Context: {args.max_seq_length}")

    # =========================================================================
    # STEP 1: Load the pre-trained model
    # =========================================================================
    # This downloads the model from HuggingFace (first time only, ~3-5 GB)
    # load_in_4bit=True means we use QLoRA: the base model is compressed to 4-bit
    # This reduces memory from ~17GB to ~10GB so it fits on consumer hardware
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,  # QLoRA: compress base model to 4-bit (saves ~60% VRAM)
    )

    # =========================================================================
    # STEP 2: Attach LoRA adapters
    # =========================================================================
    # Instead of training ALL 4 billion parameters (expensive, needs huge GPU),
    # we freeze the base model and add tiny trainable matrices called "adapters"
    # Only ~0.65% of parameters are trainable — the rest stay frozen
    # This is what makes fine-tuning possible on a laptop/free GPU
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,              # Rank: size of adapter matrices (16 = good default)
        lora_alpha=args.lora_r,     # Scaling factor (usually set equal to r)
        finetune_vision_layers=False,      # Don't train image processing layers
        finetune_language_layers=True,     # Train text understanding layers ✓
        finetune_attention_modules=True,   # Train attention (how model focuses) ✓
        finetune_mlp_modules=True,         # Train feed-forward layers ✓
    )

    # =========================================================================
    # STEP 3: Setup chat template
    # =========================================================================
    # Each model has a specific format for conversations. We must match it exactly.
    # Wrong template = gibberish output after training
    # Gemma 4 uses: <|turn>user\n...<turn|>\n<|turn>model\n...<turn|>
    # Llama 3.1 uses: <|start_header_id|>user<|end_header_id|>\n\n...
    from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
    chat_template = "gemma-4" if "gemma" in args.model.lower() else "llama-3.1"
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    # =========================================================================
    # STEP 4: Load and format our training dataset
    # =========================================================================
    # Our dataset is JSONL with conversations: [{"role": "user", ...}, {"role": "assistant", ...}]
    # We convert each conversation into the model's chat template format
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    dataset = standardize_data_formats(dataset)

    def formatting_func(examples):
        """Convert conversations to the model's expected text format."""
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)
    print(f"📚 Training on {len(dataset)} examples")

    # =========================================================================
    # STEP 5: Configure and run training
    # =========================================================================
    # SFTTrainer = Supervised Fine-Tuning Trainer (from the TRL library)
    # It handles the training loop: forward pass → compute loss → backprop → update weights
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",           # Column name containing formatted text
            per_device_train_batch_size=args.batch_size,  # Examples per step (reduce if OOM)
            gradient_accumulation_steps=4,        # Simulate larger batch: effective_batch = 2×4 = 8
            warmup_steps=10,                     # Gradually increase LR for first 10 steps
            max_steps=args.max_steps,            # Total training steps (100 ≈ 10-15 min)
            learning_rate=args.lr,               # How fast to update weights
            logging_steps=10,                    # Print loss every 10 steps
            optim="adamw_8bit",                  # 8-bit optimizer (saves VRAM)
            weight_decay=0.01,                   # Regularization (prevents overfitting)
            lr_scheduler_type="cosine",          # LR decreases smoothly over training
            seed=42,                             # Reproducibility
            output_dir=args.output_dir,          # Save checkpoints here
            save_steps=50,                       # Save checkpoint every 50 steps
            report_to="none",                    # Set to "wandb" for experiment tracking
        ),
    )

    # IMPORTANT: Only compute loss on assistant responses, NOT user messages
    # Without this, the model wastes capacity memorizing user prompts
    if "gemma" in args.model.lower():
        trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")
    else:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    # =========================================================================
    # STEP 6: Train!
    # =========================================================================
    # This is where the actual learning happens
    # You'll see loss decreasing over steps (lower = model is learning)
    # Expected: starts ~1.5-2.0, ends ~0.5-0.8 after 100 steps
    print("🏋️ Training started...")
    stats = trainer.train()
    print(f"✅ Training complete! Final loss: {stats.metrics['train_loss']:.4f}")

    # =========================================================================
    # STEP 7: Save the LoRA adapter
    # =========================================================================
    # This saves ONLY the trained adapter weights (~50-200MB)
    # NOT the full model (which is 4-9GB) — saves disk space
    # To use later: load base model + load adapter on top
    adapter_path = f"{args.output_dir}/lora-adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"💾 LoRA adapter saved → {adapter_path}")

    # =========================================================================
    # STEP 8: Export to GGUF (for Ollama deployment)
    # =========================================================================
    # GGUF = the format Ollama/llama.cpp uses to run models
    # This merges the LoRA adapter INTO the base model and quantizes to 4-bit
    # Result: a single .gguf file you can load in Ollama
    if args.export_gguf:
        gguf_path = f"{args.output_dir}/gguf"
        model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
        print(f"📦 GGUF exported → {gguf_path}")
        print(f"\n🎉 To deploy in Ollama:")
        print(f"   ollama create amazon-books -f Modelfile")
        print(f"   ollama run amazon-books 'List bestseller books of Amazon'")


if __name__ == "__main__":
    main()
