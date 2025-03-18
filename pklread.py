import pandas as pd
data = pd.read_pickle(r"C:\Users\brand\FraudDetection\dataset\2018-04-01.pkl")
print(data)  # See the first few rows
print(data.info())  # Get column names and types