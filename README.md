# Fraud Detection Project

## Overview
This repository contains a machine learning solution for detecting fraudulent transactions using a simulated dataset. The project leverages the LightGBM algorithm and custom feature engineering to classify transactions as legitimate or fraudulent based on simulated fraud patterns.

## Dataset
The dataset is a simulated collection of transaction data, including original and fraudulent transactions. It includes the following columns:
- `TRANSACTION_ID`: Unique transaction identifier
- `TX_DATETIME`: Date and time of the transaction
- `CUSTOMER_ID`: Unique customer identifier
- `TERMINAL_ID`: Unique terminal (merchant) identifier
- `TX_AMOUNT`: Transaction amount
- `TX_FRAUD`: Binary label (0 = legitimate, 1 = fraudulent)
- `TX_FRAUD_SCENARIO`: Scenario identifier for simulated fraud

### Fraud Scenarios
The fraud labels are simulated based on three scenarios:
1. **Scenario 1**: Any transaction with an amount > 220 is fraudulent (baseline pattern).
2. **Scenario 2**: Two random terminals per day have all transactions marked as fraudulent for the next 28 days (terminal compromise, e.g., phishing).
3. **Scenario 3**: Three random customers per day have 1/3 of their transactions (over the next 14 days) multiplied by 5 and marked as fraudulent (card-not-present fraud).

## Features
The model uses the following engineered features:
- Transaction amount and time-based features (`TX_AMOUNT`, `TX_TIME_SECONDS`, `hour`, `day_of_week`)
- Terminal-based features (`terminal_fraud_count`, `terminal_fraud_28d`, `terminal_fraud_28d_ratio`) to detect Scenario 2
- Customer-based features (`customer_avg_amount`, `customer_amount_14d_avg`, `amount_spike_14d`, `amount_spike_220`) to detect Scenarios 1 and 3
- Fraud rate trends and deviations

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone https://github.com/darkraider01/FraudDetection.git
   cd FraudDetection
2. **Install Dependencies**
    ```bash
    pip install requirements.txt

3. **prepare the dataset**
Place the transaction dataset .pkl files in the dataset/ subdirectory. These files are not included in the repository due to size; download them separately (e.g., from a provided source) and organize as per the original structure.

4. **run the script**
    ```bash
    python model.py

## Usage
The script trains a LightGBM model and evaluates it with multiple thresholds (0.79 to 0.84).
Output includes classification reports (precision, recall, F1-score), ROC AUC score, and feature importance.
The trained model is saved as fraud_detection_model_optimized_tuned.txt.

## Results
Best Performance: Achieved a fraud F1-score of 0.88 (threshold 0.84) and ROC AUC of 0.9857 on a test set of 350,831 transactions.
Feature Importance: Check feature_importance.png for the top contributing features (e.g., amount_spike_220, terminal_fraud_28d).

