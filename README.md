# Multimodal Return Prediction

This project explores the prediction of monthly stock returns by integrating financial fundamentals, technical indicators, and textual sentiment from 10-K reports.

## Repository Structure

ML_FOR_FINANCE_PROJECT/
- notebooks/          Development notebooks (EDA, modeling, sentiment)
- src/                Modular Python scripts for preprocessing, modeling, utils
- mapping/            Scripts for mapping between CIK, Ticker, CUSIP
- docs/               Final report
- data/               Raw and processed data (not pushed to Git)
- .gitignore
- README.md

## Setup Instructions

1. Clone the repository:

   git clone https://github.com/yourusername/ML_FOR_FINANCE_PROJECT.git  
   cd ML_FOR_FINANCE_PROJECT
   

2. Install dependencies:

   pip install -r requirements.txt

3. Add data manually inside the `data/` directory.

## Project Description

We aim to forecast monthly asset returns by combining:

- Fundamental indicators from Compustat (e.g., revenue, R&D, debt ratios)
- Technical indicators (RSI, EMA, MACD, volatility)
- Sentiment signals from MD&A sections of 10-K reports

## Models Used


## Reproducibility

Each notebook and script is modular and documented. You can reproduce experiments by running the corresponding scripts in `src/` or rerunning the notebooks in `notebooks/`.

## Authors

- Adam Chahed Ouazzani
- Amine Bengelloun 
- Quentin Brian Rossier 
- Ayman Bakiri 
- Amine Boucetta 
