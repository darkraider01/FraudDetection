import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import warnings
import sys
try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
warnings.filterwarnings('ignore')

# Open a log file to capture all output
with open('model_output.log', 'w') as log_file:
    # Redirect stdout to log file
    original_stdout = sys.stdout
    sys.stdout = log_file

    # 1. Load and Combine Data
    folder_path = 'c:/Users/brand/FraudDetection/dataset/'  # Update this path
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    data_list = [pd.read_pickle(os.path.join(folder_path, f)) for f in pkl_files]
    data = pd.concat(data_list, ignore_index=False)  # Preserve original indices

    print(f"Loaded {len(data)} transactions from {len(pkl_files)} files.")
    print("Initial columns:", data.columns.tolist())  # Debug: Check initial columns
    print("Initial TX_DATETIME head:", data['TX_DATETIME'].head())  # Debug: Check raw values
    print("Initial TX_DATETIME type:", data['TX_DATETIME'].dtype)  # Debug: Check datetime type

    # Ensure TX_DATETIME is datetime type and drop invalid rows in-place
    data['TX_DATETIME'] = pd.to_datetime(data['TX_DATETIME'], errors='coerce')
    data.dropna(subset=['TX_DATETIME'], inplace=True)  # In-place drop to avoid copy
    print("Data cleaned, shape:", data.shape)  # Debug: Verify shape after cleaning

    # 2. Feature Engineering
    data['TX_TIME_SECONDS'] = pd.to_numeric(data['TX_TIME_SECONDS'], errors='coerce')
    data['TX_TIME_DAYS'] = pd.to_numeric(data['TX_TIME_DAYS'], errors='coerce')

    # Remove any initial duplicate columns, preserving datetime TX_DATETIME
    data = data.loc[:, ~data.columns.duplicated(keep='first')]
    print("Columns after removing duplicates:", data.columns.tolist())  # Debug: Verify unique columns
    print("TX_DATETIME type after duplicates removed:", data['TX_DATETIME'].dtype)  # Debug: Check type

    # Sort by datetime for rolling features
    data = data.sort_values(by='TX_DATETIME', kind='mergesort')
    data['TX_DATETIME'] = pd.to_datetime(data['TX_DATETIME'])  # Re-enforce datetime type
    print("Data sorted by TX_DATETIME, first few rows:", data[['TX_DATETIME']].head())  # Debug: Verify sorting
    print("TX_DATETIME type after sorting:", data['TX_DATETIME'].dtype)  # Debug: Check type

    # Terminal features
    data['terminal_fraud_count'] = data.groupby('TERMINAL_ID')['TX_FRAUD'].cumsum() - data['TX_FRAUD']
    data['terminal_tx_count'] = data.groupby('TERMINAL_ID').cumcount()
    data['terminal_fraud_rate'] = data['terminal_fraud_count'] / (data['terminal_tx_count'] + 1)
    print("Terminal features added, first few rows:", data[['TX_DATETIME', 'terminal_tx_count']].head())  # Debug: Verify terminal features

    # Customer features
    data['customer_avg_amount'] = data.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform('mean')
    data['amount_deviation'] = data['TX_AMOUNT'] / (data['customer_avg_amount'] + 1)
    data['customer_fraud_count'] = data.groupby('CUSTOMER_ID')['TX_FRAUD'].cumsum() - data['TX_FRAUD']

    # Time features
    data['hour'] = data['TX_DATETIME'].dt.hour
    data['day_of_week'] = data['TX_DATETIME'].dt.dayofweek
    data['TX_DATETIME'] = pd.to_datetime(data['TX_DATETIME'])  # Re-enforce datetime type
    print("Time features added, TX_DATETIME type:", data['TX_DATETIME'].dtype)  # Debug: Check type

    # Enhanced rolling features with correct time handling
    print("Starting rolling feature computation...")
    # Sort by TERMINAL_ID and TX_DATETIME for rolling
    data = data.sort_values(by=['TERMINAL_ID', 'TX_DATETIME'])
    data['TX_DATETIME'] = pd.to_datetime(data['TX_DATETIME'])  # Re-enforce datetime type
    print("Data sorted for rolling, TX_DATETIME type:", data['TX_DATETIME'].dtype)  # Debug: Check type

    # Terminal rolling features using TX_DATETIME as time axis (28D for Scenario 2)
    temp_terminal = data.groupby('TERMINAL_ID').apply(
        lambda x: x.sort_values('TX_DATETIME')
                  .drop_duplicates(subset='TX_DATETIME', keep='first')
                  .rolling('28D', on='TX_DATETIME')['TX_FRAUD'].sum().shift(1)
    )
    temp_terminal = temp_terminal.reset_index(level=0, drop=True)  # Remove MultiIndex level
    data = data.assign(terminal_fraud_28d=temp_terminal).fillna({'terminal_fraud_28d': 0})
    data['TX_DATETIME'] = pd.to_datetime(data['TX_DATETIME'])  # Re-enforce datetime type
    print("Terminal fraud 28d computed, columns:", data.columns.tolist())  # Debug: Verify columns

    # Compute rolling counts directly
    temp_counts = data.groupby('TERMINAL_ID').apply(
        lambda x: x.sort_values('TX_DATETIME')
                  .drop_duplicates(subset='TX_DATETIME', keep='first')
                  .rolling('28D', on='TX_DATETIME')['terminal_tx_count'].count()
    )
    temp_counts = temp_counts.reset_index(level=0, drop=True)  # Remove MultiIndex level
    data = data.assign(rolling_counts=temp_counts)
    data['terminal_fraud_28d_ratio'] = data['terminal_fraud_28d'] / data['rolling_counts'].shift(1).fillna(1)
    data.drop(columns=['rolling_counts'], inplace=True)
    data['TX_DATETIME'] = pd.to_datetime(data['TX_DATETIME'])  # Re-enforce datetime type
    print("Terminal fraud 28d ratio computed, first few values:", data[['TX_DATETIME', 'terminal_fraud_28d_ratio']].head())  # Debug: Verify ratio

    # Customer rolling features (14D for Scenario 3)
    temp_customer_14d = data.groupby('CUSTOMER_ID').apply(
        lambda x: x.sort_values('TX_DATETIME')
                  .drop_duplicates(subset='TX_DATETIME', keep='first')
                  .rolling('14D', on='TX_DATETIME')['TX_AMOUNT'].mean().shift(1)
    )
    temp_customer_14d = temp_customer_14d.reset_index(level=0, drop=True)  # Remove MultiIndex level
    data = data.assign(customer_amount_14d_avg=temp_customer_14d).fillna({'customer_amount_14d_avg': data['TX_AMOUNT'].mean()})

    # Compute amount spikes and flags based on different averages
    data['amount_spike'] = data['TX_AMOUNT'] / (data['customer_avg_amount'] + 1)
    data['amount_spike_flag'] = (data['amount_spike'] > 5).astype(int)  # Detects 5x spikes from Scenario 3
    data['amount_spike_14d'] = data['TX_AMOUNT'] / (data['customer_amount_14d_avg'] + 1)
    data['amount_spike_flag_14d'] = (data['amount_spike_14d'] > 5).astype(int)  # Specific to 14D window
    data['amount_spike_220'] = (data['TX_AMOUNT'] > 220).astype(int)  # Explicitly flags Scenario 1

    data['amount_deviation'] = data['TX_AMOUNT'] / (data['customer_avg_amount'] + 1)
    data['amount_deviation_14d'] = data['TX_AMOUNT'] / (data['customer_amount_14d_avg'] + 1)

    data['TX_DATETIME'] = pd.to_datetime(data['TX_DATETIME'])  # Re-enforce datetime type
    print("Customer features computed, columns:", data.columns.tolist())  # Debug: Verify columns

    # Remove duplicate TX_DATETIME if any
    data = data.loc[:, ~data.columns.duplicated(keep='first')]
    print("Columns after final deduplication:", data.columns.tolist())  # Debug: Verify unique columns

    # Add fraud rate trend
    data['terminal_fraud_trend'] = data.groupby('TERMINAL_ID')['terminal_fraud_rate'].diff().fillna(0)

    # 3. Prepare Features and Target
    le = LabelEncoder()
    data['CUSTOMER_ID'] = le.fit_transform(data['CUSTOMER_ID'])
    data['TERMINAL_ID'] = le.fit_transform(data['TERMINAL_ID'])

    features = [
        'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS', 
        'CUSTOMER_ID', 'TERMINAL_ID', 'hour', 'day_of_week',
        'terminal_fraud_count', 'terminal_fraud_rate', 'terminal_tx_count',
        'customer_avg_amount', 'amount_deviation', 'amount_deviation_14d',
        'amount_spike', 'amount_spike_14d', 'amount_spike_220',
        'amount_spike_flag', 'amount_spike_flag_14d',
        'customer_fraud_count', 'terminal_fraud_28d', 'terminal_fraud_28d_ratio',
        'customer_amount_14d_avg', 'terminal_fraud_trend'
    ]
    X = data[features].fillna(0)  # Fill NaN from shifts
    y = data['TX_FRAUD']

    print("Features prepared, shape:", X.shape)  # Debug: Check feature matrix

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split, train shape:", X_train.shape, "test shape:", X_test.shape)  # Debug: Check split

    # 5. Train LightGBM Model
    scale_pos_weight = (y == 0).sum() / (y == 1).sum() / 2.5
    print("Scale pos weight:", scale_pos_weight)  # Debug: Check class balance

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1
    }

    print("Starting model training...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_test],
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )
    print("Model training completed.")

    # 6. Predict and Evaluate with Multiple Thresholds
    thresholds = [round(0.79 + 0.01 * i, 2) for i in range(6)]  # [0.79, 0.80, 0.81, 0.82, 0.83, 0.84]
    for threshold in thresholds:
        y_pred = (model.predict(X_test) > threshold).astype(int)
        print(f"\nClassification Report (Threshold = {threshold}):")
        print(classification_report(y_test, y_pred))
        print(f"Threshold {threshold} processed successfully.")  # Debug: Confirm each threshold is evaluated
        log_file.flush()  # Ensure output is written to file immediately

    print(f"ROC AUC Score: {roc_auc_score(y_test, model.predict(X_test)):.4f}")

    # 7. Feature Importance
    if PLOT_AVAILABLE:
        ax = lgb.plot_importance(model, max_num_features=10, figsize=(10, 6))
        plt.savefig('feature_importance.png')  # Save plot to file
        plt.show()
        print("Feature importance plot saved as 'feature_importance.png' and displayed.")
    else:
        importance = model.feature_importance()
        feature_names = model.feature_name()
        for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{name}: {imp}")

    # 8. Save Model
    model.save_model('fraud_detection_model_optimized_tuned.txt')
    print("Model saved as 'fraud_detection_model_optimized_tuned.txt'")

    print("Script execution completed successfully.")
    sys.stdout = original_stdout  # Restore original stdout
    with open('model_output.log', 'a') as log_file:
        print("Full output saved to model_output.log. Please check the file for complete results.", file=log_file)
