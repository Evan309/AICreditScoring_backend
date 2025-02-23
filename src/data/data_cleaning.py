import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler

mlb = MultiLabelBinarizer()
scaler = StandardScaler()

def process_data(data_path):

    df = pd.read_csv(data_path, dtype={"Monthly_Balance" : "str"})

    # drop ID, Customer_ID, Name, SSN, Month
    df = df.drop(columns = ["ID", "Customer_ID", "Name", "SSN", "Month"])

    # drop any duplicate rows
    df.drop_duplicates(inplace=True)

    # strip values of excess characters
    def strip_values(column_names):
        for col in column_names:
            if col != "Occupation":
                df[col] = df[col].astype(str).str.strip("_").replace({"nan": np.nan, "": np.nan})

    strip_values(df.columns)

    # drop na values
    df["Occupation"] = df["Occupation"].replace({"_______" : "Other"})
    df["Num_Bank_Accounts"] = df["Num_Bank_Accounts"].replace({-1 : np.nan})
    df["Num_of_Loan"] = np.where(df["Num_of_Loan"].astype(int) < 0, 0, df['Num_of_Loan'])
    df["Delay_from_due_date"] = np.where(df["Delay_from_due_date"].astype(int) < 0, np.nan, df["Delay_from_due_date"])
    df["Credit_Mix"] = df["Credit_Mix"].replace({"" : np.nan})
    df["Payment_Behaviour"] = df["Payment_Behaviour"].replace({"!@9#%8" : np.nan})
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace({"NM" : np.nan})
    df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].replace({"_" : np.nan})
    df["Amount_invested_monthly"] = df["Amount_invested_monthly"].replace({"__10000__" : 10000.00})
    df.replace("nan", np.nan, inplace=True)

    df.dropna(inplace=True)

    # change age in years to age in days
    def year_to_month(t):
        t = int(t)
        return t * 12

    df["Age"] = df["Age"].apply(year_to_month)

    # change time in string to time in days number
    def string_to_months(s):
        s = s.replace(" and", "")
        s = s.split(" ")
        num_months = int(s[0]) * 12 + int(s[2])
        return num_months

    df["Credit_History_Age"] = df["Credit_History_Age"].apply(string_to_months)

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

    # ordinal encode credit mix
    credit_mix_mapping = {"Bad": 0, "Standard": 1, "Good": 2}
    df["Credit_Mix"] = df["Credit_Mix"].map(credit_mix_mapping)

    # encode Payment of min amount
    yes_no_mapping = {"No": 0, "Yes": 1}
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].map(yes_no_mapping)

    # one-hot encode payment behaviour
    def parse_payment_behaviour(s):
        parts = s.split('_')
        spend = parts[0]
        size = parts[2]
        return spend, size

    df[['Spend_Level', 'Payment_Size']] = df['Payment_Behaviour'].apply(lambda s: pd.Series(parse_payment_behaviour(s)))

    spend_mapping = {"Low": 0, "High": 1}
    size_mapping = {"Small": 0, "Medium": 1, "Large": 2}
    df["Spend_Level"] = df["Spend_Level"].map(spend_mapping)
    df["Payment_Size"] = df["Payment_Size"].map(size_mapping)
    df.drop(columns=["Payment_Behaviour"], inplace=True)

    # if train data then encode credit scores
    if "Credit_Score" in df.columns:
        credit_score_mapping = {"Poor": 0, "Standard": 1, "Good": 2}
        df["Credit_Score"] = df["Credit_Score"].map(credit_score_mapping)

    #convert all values to float
    df = df.astype(float)
    return df

def standardize_data(df):
    standardized_data = df

    num_columns = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", 
                   "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", 
                   "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age", 
                   "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
                   "Changed_Credit_Limit", "Num_of_Delayed_Payment"]
    
    standardized_data[num_columns] = scaler.fit_transform(df[num_columns])

    # shift data up by min value so values are positive
    for col in num_columns:
        min_value = standardized_data[col].min()
        if min_value < 0:
            shift_value = abs(min_value) + 1
            standardized_data[col] += shift_value
    
    return standardized_data



df_train = process_data("/Users/evanshi/Desktop/Personal-Projects/AICreditScoring/data/raw/train.csv")
df_test = process_data("/Users/evanshi/Desktop/Personal-Projects/AICreditScoring/data/raw/test.csv")

standardized_train = standardize_data(df_train)
standardized_test = standardize_data(df_test)

df_train.to_csv("/Users/evanshi/Desktop/Personal-Projects/AICreditScoring/data/processed/processed_train.csv", index=False)
df_test.to_csv("/Users/evanshi/Desktop/Personal-Projects/AICreditScoring/data/processed/processed_test.csv", index=False)