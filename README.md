# **Multimodal Return Prediction: Merging Fundamentals, Technicals, and Financial Texts**

## **Overview**

This project investigates the use of multiple data modalities to forecast monthly stock returns. Specifically, we combine structured financial indicators, engineered technical features, and sentiment extracted from 10-K filings (MD&A section) using FinBERT. The task is framed as a binary classification problem: predicting whether a firm's return in the following month will be positive or negative.

We explore a wide range of models, from random forests and gradient boosting to recurrent neural networks and transformer-based architectures for time series.

---

## **Repository Philosophy**

This repository was designed with clarity, modularity, and reproducibility in mind. We aimed to create a clean and understandable codebase that is easy to navigate and build upon.

- **Clear folder structure**: Each part of the pipeline (data, preprocessing, modeling, evaluation) is logically separated.
- **Modular and documented code**: Python scripts are written for reuse, and notebooks illustrate how to run end-to-end experiments.
- **Reproducible workflows**: All major experiments are either scripted or runnable in Jupyter notebooks, with fixed random seeds where relevant.
- **Lightweight setup**: The number of dependencies is minimized and documented in `requirements.txt`.

---

## **Repository Structure**

```
ML_FOR_FINANCE_PROJECT/
├── notebooks/          # Development notebooks: EDA, model training, sentiment extraction
├── src/                # Modular Python scripts (data processing, models, utils)
├── mapping/            # Scripts to map CIK, Ticker, CUSIP identifiers
├── docs/               # Dataset guides, preprocessing instructions
├── Report/             # Final LaTeX report and bibliography
├── data/               # Raw and processed data (not tracked in Git)
├── requirements.txt    # Python dependencies
└── README.md           # Project overview and instructions
```

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ML_FOR_FINANCE_PROJECT.git
   cd ML_FOR_FINANCE_PROJECT
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download and place all necessary raw datasets (Compustat, CRSP, 10-Ks) into the `data/` folder.

---

## **Models and Methods**

### **1. Random Forest Classifier**
- Input: Flattened firm-month tabular data.
- Preprocessing: Median imputation, standardization.
- Output: Binary prediction of positive return.

### **2. XGBoost Classifier**
- Objective: Binary logistic loss optimized via AUC.
- Strength: Handles feature sparsity and imbalance well.
- Hyperparameter tuning: Tree depth, learning rate, estimators.

### **3. Random Forest Regressor**
- Objective: Estimate raw return value before thresholding into binary labels.
- Metrics: MAE, MSE, R².

### **4. LSTM-Based Neural Networks**
- Windowed modeling over 6-month rolling firm-level inputs.
- Variants:
  - `SmallLSTM`: Lightweight model for basic sequence learning.
  - `LargeLSTM`: Deeper, bidirectional architecture with dropout.
  - `EnhancedLSTM`: Adds LayerNorm and a deeper non-linear head.

### **5. Transformer and InceptionTime**
- `StockTransformer`: Temporal transformer over monthly firm data.
- `InceptionTime`: CNN-based time series model with residuals and multi-scale convolutions.

### **6. FinBERT-Based Sentiment Features**
- Extracted from MD&A sections of 10-K reports.
- Sentiment classification via FinBERT with chunking and aggregation.
- Integrated as an additional categorical feature in all models.

---

## **Usage**

All scripts are modular and run independently. Key pipelines:

- Train Random Forest:
  ```bash
  python src/rf_classifier.py
  ```

- Train LSTM or Transformer:
  ```bash
  python training.ipynb
  ```

- Generate sentiment scores:
  ```bash
  python src/sentiment_analysis.py
  ```

Ensure all datasets are properly preprocessed and aligned before launching model training.

---

## **Reproducibility**

- All preprocessing and modeling steps are documented in `notebooks/` and implemented in `src/`.
- Random seeds are fixed for deterministic runs.
- Chronological splits ensure no data leakage.

---

## **Contributors**

- Adam Chahed Ouazzani  
- Amine Bengelloun  
- Quentin Brian Rossier  
- Ayman Bakiri  
- Amine Boucetta
```
