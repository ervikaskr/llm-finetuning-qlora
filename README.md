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
ollama create amazon-books -f Modelfile
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
| Base Model | Gemma 4 E4B (4B params) | Small, multimodal, Apache 2.0 |
| Method | QLoRA (4-bit + LoRA r=16) | Fits on 10GB VRAM / 36GB Mac |
| Framework | Unsloth + TRL | 2x faster, 60% less memory |
| Dataset | 500 books → 1050 Q&A pairs | Diverse question types |
| Export | GGUF Q4_K_M | For Ollama local deployment |
| Training Time | ~15 min (Mac M3) / ~8 min (T4) | 100 steps |

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
# Full pipeline (6 commands)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train.py --max_steps 100
python scripts/evaluate.py --query "List bestseller books of Amazon"
```

## 📝 License

MIT — Dataset from [Kaggle](https://www.kaggle.com/datasets/shambhurajejagadale/amazon-bestselling-books-dataset-500-books).
