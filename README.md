# **Multimodal Return Prediction: Merging Fundamentals, Technicals, and Financial Texts**

## **Overview**

This project investigates whether combining multiple data modalities—structured financial features, technical indicators, and textual sentiment—can improve monthly stock return prediction. In particular, we extract sentiment from the MD&A section of 10-K filings using FinBERT and integrate it with firm-level financial and price-based data.

The prediction task is framed as a binary classification problem: will a firm's return be positive or negative in the next month? We evaluate a wide spectrum of models, including random forests, gradient boosting, recurrent neural networks, transformers, and multi-scale convolutional networks.

---

## **Repository Philosophy**

The repository is structured for clarity, reproducibility, and collaboration. Key design principles include:

- **Modular architecture**: All scripts are separated by function (data handling, modeling, evaluation).
- **Reproducible experiments**: Major pipelines are either executable scripts or Jupyter notebooks.
- **Minimal setup**: Dependencies are listed in `requirements.txt`.
- **Clean organization**: Easy navigation and separation of raw data, code, notebooks, and documentation.

---

## **Repository Structure**

```text
ML_FOR_FINANCE_PROJECT/
├── docs/                # Documentation and guides
├── mapping/             # Scripts for identifier mapping (CIK, CUSIP, Ticker)
├── notebooks/EDA/       # Exploratory analysis and visualizations
├── src/                 # Core Python modules: preprocessing, modeling, sentiment
├── .gitignore           # Git exclusions
├── README.md            # This file
├── requirements.txt     # Dependencies
```

> - `data/`: Contains raw and processed datasets (not included in Git)
> - `Report/`: Contains the full LaTeX report and bibliography

---

## **Installation**

Clone the repository and set up the environment:

```bash
git clone https://github.com/yourusername/ML_FOR_FINANCE_PROJECT.git
cd ML_FOR_FINANCE_PROJECT
```

Install required packages:

```bash
pip install -r requirements.txt
```

You must manually download and place all raw datasets (Compustat, CRSP, and 10-K reports) into the `data/` directory.

---

## **Models and Methods**

### **1. Random Forest Classifier**
- Uses tabular financial and technical data.
- Preprocessing includes standardization and median imputation.
- Baseline model for monthly return direction classification.

### **2. XGBoost Classifier**
- Trained to maximize ROC AUC.
- Tuned over tree depth, learning rate, and class imbalance parameters.
- Demonstrates class skew under default thresholds.

### **3. Recurrent Neural Networks (LSTM variants)**
- Models receive rolling windows (6 or 24 months) of firm histories.
- Architectures:
  - `SmallLSTM`: Lightweight and shallow.
  - `LargeLSTM`: Two-layer bidirectional with dropout.
  - `EnhancedLSTM`: Adds LayerNorm and a dense classification head.

### **4. Transformer-Based Classifier**
- Applies self-attention over temporal sequences.
- Captures long-range dependencies more effectively than recurrence.

### **5. InceptionTime**
- Uses 1D convolutions with multiple kernel sizes and residual blocks.
- Best performing model across metrics: accuracy, ROC AUC, and F1.

### **6. FinBERT Sentiment Features**
- Sentiment extracted from MD&A sections of 10-Ks using FinBERT.
- Texts are split into overlapping chunks and aggregated via majority vote.
- Sentiment used as a categorical feature in tree-based models.
- Not yet integrated into neural networks due to limited firm coverage. This remains a direction for future work.

---

## **Usage**

Train and evaluate models as follows:

```bash
# Train Random Forest
python src/rf_classifier.py

# Train LSTM or Transformer (via notebook)
jupyter notebook notebooks/EDA/training.ipynb

# Generate sentiment labels
python src/sentiment_analysis.py
```

Ensure that datasets are preprocessed and aligned before training.


---

## **Contributors**

- Adam Chahed Ouazzani  
- Amine Bengelloun  
- Quentin Brian Rossier  
- Ayman Bakiri  
- Amine Bousseta
```
