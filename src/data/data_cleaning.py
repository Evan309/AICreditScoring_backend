import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
df = pd.read_csv("/Users/evanshi/Desktop/Personal-Projects/AICreditScoring/data/raw/test.csv")

# drop ID, Customer_ID, Name, SSN, Month
df = df.drop(columns = ["ID", "Customer_ID", "Name", "SSN", "Month"])

# drop any duplicate rows
df.drop_duplicates(inplace=True)

# drop na values
df["Occupation"] = df["Occupation"].replace({"_______" : "Other"})
df["Payment_Behaviour"] = df["Payment_Behaviour"].replace({"!@9#%8" : np.nan})
df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace({"NM" : np.nan})
df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].replace({"_" : np.nan})
df["Amount_invested_monthly"] = df["Amount_invested_monthly"].replace({"__10000__" : 10000.00})
df.dropna(inplace=True)

# change age in years to age in days
def year_to_age(t):
    t = int(t)
    return t * 365

df["Age"] = df["Age"].str.replace("_", "", regex=False)
df["Age"] = df["Age"].apply(year_to_age)

# one-hot encode occupation types
occupation_df = pd.get_dummies(df["Occupation"], prefix="Occ")
occupation_df = occupation_df.astype(int)
df = pd.concat([df, occupation_df], axis=1)
df.drop(columns=["Occupation"], inplace=True)

# one-hot encode loan types
def clean_loan_type_string(str):
    str = str.replace("and", "")
    loans = [loan.strip() for loan in str.split(",") if loan.strip() != ""]
    return list(set(loans))

df["Type_of_Loan"] = df["Type_of_Loan"].apply(clean_loan_type_string)

encoded_loan = mlb.fit_transform(df["Type_of_Loan"])
loan_df = pd.DataFrame(encoded_loan, columns=mlb.classes_, index=df.index)
df = pd.concat([df, loan_df], axis=1)
df.drop(columns=["Type_of_Loan"], inplace=True)

print(df["Credit_History_Age"])