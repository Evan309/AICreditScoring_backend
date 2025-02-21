import pandas as pd
import numpy as np

df = pd.read_csv("/Users/evanshi/Desktop/Personal-Projects/AICreditScoring/data/raw/test.csv")

# drop ID, Customer_ID, Name, SSN, Month
# does not effect classification in Neural Network
df = df.drop(columns = ["ID", "Customer_ID", "Name", "SSN", "Month"])

# change column names to lower case
new_column_names = [name.lower() for name in df.columns]
df.columns = new_column_names

# drop any duplicate rows
df.drop_duplicates(inplace=True)

# drop na values
df.dropna(inplace=True)
df["occupation"] = df["occupation"].replace({"_______" : "Other"})
df["payment_behaviour"] = df["payment_behaviour"].replace({"!@9#%8" : np.nan})
df.dropna(inplace=True)

print(df)