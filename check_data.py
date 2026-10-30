import pandas as pd

# Load the dataset
df = pd.read_csv("data/SMSSpamCollection.csv", encoding="latin-1")

# Show the first 5 rows
print(df.head())

# Show the column names
print("\nColumns:")
print(df.columns)

# Show the number of rows and columns
print("\nDataset shape:")
print(df.shape)