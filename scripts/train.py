"""
Fine-Tune Gemma 4 E4B on Amazon Bestselling Books
==================================================
Supports: Local (Mac M-series / NVIDIA GPU) and Kaggle/Colab

Usage:
  python scripts/train.py                    # Default: 100 steps
  python scripts/train.py --max_steps 200    # Custom steps
  python scripts/train.py --model unsloth/Llama-3.1-8B-Instruct  # Different model
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on Amazon Books")
    parser.add_argument("--model", default="unsloth/gemma-4-E4B-it", help="Model name")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_dir", default="./outputs/model")
    parser.add_argument("--export_gguf", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(__file__).parent.parent / "data" / "train.jsonl"

    if not data_path.exists():
        print("❌ Training data not found. Run: python scripts/prepare_data.py")
        return

    print(f"🚀 Fine-tuning {args.model}")
    print(f"   Steps: {args.max_steps}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"   LoRA rank: {args.lora_r}, Context: {args.max_seq_length}")

    # 1. Load model
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    # 2. Attach LoRA
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    # 3. Setup chat template
    from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
    chat_template = "gemma-4" if "gemma" in args.model.lower() else "llama-3.1"
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    # 4. Load dataset
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    dataset = standardize_data_formats(dataset)

    def formatting_func(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)
    print(f"📚 Training on {len(dataset)} examples")

    # 5. Train
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=args.output_dir,
            save_steps=50,
            report_to="none",
        ),
    )

    # Train only on assistant responses
    if "gemma" in args.model.lower():
        trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")
    else:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    print("🏋️ Training started...")
    stats = trainer.train()
    print(f"✅ Training complete! Loss: {stats.metrics['train_loss']:.4f}")

    # 6. Save adapter
    adapter_path = f"{args.output_dir}/lora-adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"💾 LoRA adapter saved → {adapter_path}")

    # 7. Export GGUF
    if args.export_gguf:
        gguf_path = f"{args.output_dir}/gguf"
        model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
        print(f"📦 GGUF exported → {gguf_path}")
        print(f"\n🎉 To run in Ollama:")
        print(f"   ollama create amazon-books -f Modelfile")
        print(f"   ollama run amazon-books 'List bestseller books of Amazon'")


if __name__ == "__main__":
    main()
