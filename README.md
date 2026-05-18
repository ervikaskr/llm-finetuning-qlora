# 📚 Amazon Books QA — LLM Fine-Tuning Project

Fine-tune Gemma 4 (or any open LLM) to answer questions about Amazon's top 500 bestselling books using QLoRA.

**Before fine-tuning:** Model says "I don't have real-time access to Amazon's data..."  
**After fine-tuning:** Model lists actual bestsellers with titles, authors, ratings, and prices.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Kaggle Dataset (500 books CSV)                             │
│       ↓                                                     │
│  prepare_data.py → 1050 Q&A pairs (JSONL)                  │
│       ↓                                                     │
│  train.py (Unsloth + QLoRA)                                │
│       ↓                                                     │
│  LoRA Adapter (~50MB) + GGUF Export                         │
│       ↓                                                     │
│  Ollama (local deployment) → Ask questions!                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Setup
```bash
git clone https://github.com/ervikaskr/llm-finetuning-qlora.git
cd llm-finetuning-qlora
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download & Prepare Data
```bash
python scripts/download_data.py
python scripts/prepare_data.py
```

### 3. Train (choose one)

**Option A: Local (Mac M-series / NVIDIA GPU)**
```bash
python scripts/train.py --max_steps 100
```

**Option B: Kaggle (free GPU)**
- Upload `notebooks/train_kaggle.ipynb` to Kaggle
- Enable GPU T4 x2 → Run all cells
- Download GGUF from Output tab

### 4. Deploy & Test
```bash
# Fuse adapter into base model + export GGUF
mlx_lm.fuse \
  --model mlx-community/gemma-2-2b-it-4bit \
  --adapter-path ./outputs/adapter \
  --export-gguf \
  --gguf-path ./outputs/model.gguf

# Load in Ollama
ollama create amazon-books -f Modelfile

# Test it!
ollama run amazon-books "List bestseller books of Amazon"
```

## 📂 Project Structure

```
llm-finetuning-qlora/
├── README.md
├── requirements.txt
├── Modelfile                    # Ollama deployment config
├── .gitignore
├── data/
│   └── Amazon_BestSelling_Books_500.csv  # Raw dataset
├── scripts/
│   ├── download_data.py         # Download from Kaggle
│   ├── prepare_data.py          # CSV → JSONL training pairs
│   ├── train.py                 # Fine-tuning (local/cloud)
│   └── evaluate.py              # Test & evaluate model
├── notebooks/
│   └── train_kaggle.ipynb       # Kaggle notebook (free GPU)
├── configs/                     # (optional) hyperparameter configs
└── outputs/                     # (gitignored) model artifacts
    └── model/
        ├── lora-adapter/        # LoRA weights (~50MB)
        └── gguf/                # Quantized model for Ollama
```

## 🔧 Technical Details

| Component | Choice | Why |
|-----------|--------|-----|
| Base Model | Gemma 2 2B (4-bit MLX) | Small, fast, fits any Mac |
| Method | LoRA via MLX | Native Apple Silicon, fastest on Mac |
| Framework | Apple MLX (mlx-lm) | No CUDA needed, uses unified memory |
| Dataset | 500 books → 1050 Q&A pairs | Diverse question types |
| Export | GGUF via mlx_lm.convert | For Ollama local deployment |
| Training Time | ~10-15 min (M3 Pro) | 100 iterations |

## 📊 Results

| Metric | Before | After |
|--------|--------|-------|
| "List bestsellers" | ❌ "I don't have access..." | ✅ Lists actual top 10 books |
| Book-specific Q&A | ❌ Generic/hallucinated | ✅ Correct title, author, rating |
| Genre recommendations | ❌ Vague suggestions | ✅ Specific books with details |

## 🧠 What I Learned

- **QLoRA** makes fine-tuning accessible on consumer hardware (only 0.65% of params trained)
- **Data quality > quantity** — 1050 well-structured pairs outperform 10K noisy ones
- **Chat template matters** — wrong template = gibberish output
- **train_on_responses_only** prevents the model from memorizing user prompts
- **GGUF export** enables deployment anywhere Ollama runs (Mac, Linux, Windows)

## 🔄 Reproduce

```bash
# Full pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train.py --iters 100
mlx_lm.fuse --model mlx-community/gemma-2-2b-it-4bit --adapter-path ./outputs/adapter --export-gguf --gguf-path ./outputs/model.gguf
ollama create amazon-books -f Modelfile
ollama run amazon-books "List bestseller books of Amazon"
```

## 📝 License

MIT — Dataset from [Kaggle](https://www.kaggle.com/datasets/shambhurajejagadale/amazon-bestselling-books-dataset-500-books).
