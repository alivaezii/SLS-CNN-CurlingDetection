# CNNPROJECT — Curling Detection in SLS with Lightweight CNN

This repository contains the code and analysis for detecting **curling defects** in the Selective Laser Sintering (SLS) process using a lightweight convolutional neural network (CNN).  
The project emphasizes **anti-leakage dataset handling**, reproducible evaluation, and transparent reporting.

---

## 🔍 Overview

- **Goal**: Develop a lightweight, CPU-friendly CNN (MobileNetV3-small) for binary classification (`curling` vs. `ohne_curling`).  
- **Key Focus**:  
  - Strict anti-leakage pipeline  
  - Threshold tuning on validation set  
  - Robust evaluation on an unseen test set  
  - Transparency in dataset handling under data protection constraints  

---

## 📂 Repository Structure

```
cnnproject/
│
├── README.md                # Project description (this file)
├── requirements.txt         # Pip dependencies
├── environment.yml          # Conda environment definition
├── .gitignore               # Ignore large/unnecessary files
│
├── data/
│   ├── manifests/           # SHA256 + size based manifests (no raw data)
│   │   ├── manifest.tsv
│   │   ├── train_manifest.tsv
│   │   ├── val_manifest.tsv
│   │   └── test_manifest.tsv
│   └── README.md            # Note on data availability & Datenschutz
│
├── scripts/                 # Core Python scripts
│   ├── spliter.py           # Leakage-safe dataset splitting
│   ├── leakagecheck.py      # Duplicate / leakage detection
│   ├── train.py             # Model training
│   ├── eval_threshold.py    # Threshold tuning on validation set
│   ├── eval_test.py         # Final evaluation on test set
│   └── utils.py             # Shared helper functions
│
├── results/                 # Evaluation outputs
│   ├── val_threshold.json
│   ├── test_report.json
│   ├── plots/
│   │   ├── pr_curve.png
│   │   ├── roc_curve.png
│   │   └── confusion_matrix.png
│   └── README.md
│
├── notebooks/               # Interactive analysis
│   └── analysis.ipynb
│
└── docs/                    # Reports and figures
    └── interim_report.pdf
```

---

## ⚙️ Setup

### Using pip
```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Using conda
```bash
conda env create -f environment.yml
conda activate cnnproject
```

---

## 🚀 Usage

1. **Check for leakage**  
   ```bash
   python scripts/leakagecheck.py
   ```

2. **Train the model**  
   ```bash
   python scripts/train.py
   ```

3. **Tune threshold on validation set**  
   ```bash
   python scripts/eval_threshold.py
   ```

4. **Evaluate on test set**  
   ```bash
   python scripts/eval_test.py
   ```

Results (JSON + plots) will be written into `results/`.

---

## 📊 Current Results

- **Dataset splits**:  
  - Train: 43,524 images  
  - Validation: 31,900 images  
  - Test: 772 images (57 positives, 715 negatives)

- **Leakage check**: No duplicates across splits (SHA256, filename+size).  
- **Test performance**:  
  - Accuracy: 100%  
  - Precision/Recall/F1 (positive class): 100%  
  - Confusion Matrix: `[[715, 0], [0, 57]]`

---

## 📁 Data Availability

Due to **Datenschutz (data protection regulations)**, the raw SLS image dataset **cannot be published** in this repository.  

To ensure reproducibility and scientific transparency, we provide:
- **Complete manifests** (`manifest.tsv`, `train/val/test manifests`) with file names, sizes, and SHA256 hashes for integrity verification.  
- **All scripts** used to generate the splits, check for leakage, and run training/evaluation.  
- **Evaluation outputs** (`val_threshold.json`, `test_report.json`, plots) for validation and test sets.  

This ensures the pipeline is **fully reproducible** while respecting data protection requirements.

---

## 📌 Next Steps

- Perform **K-fold cross-validation** on the full dataset.  
- Conduct **robustness tests** (noise, rotation, blur).  
- Compare with **simpler baselines** (e.g., logistic regression, SVM).  
- Evaluate on **new unseen data** if available.

---

## 📜 License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Ali Vaezi**  
PhD Candidate — FH Campus Wien  
Email: *[your-email]*  
