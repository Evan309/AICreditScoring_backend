import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import logging
import os

# retrieve file path from env
load_dotenv()
train_file_path = os.getenv("TRAIN_FILE_PATH")
test_file_path = os.getenv("TEST_FILE_PATH")

# Initialize preprocessors
mlb = MultiLabelBinarizer()
scaler = StandardScaler()

# Configure logging
logging.basicConfig(level=logging.INFO)

def strip_values(df, column_names):
    """Strips excess characters and replaces specified values in the given columns."""
    for col in column_names:
        if col != "Occupation":
            df[col] = df[col].astype(str).str.strip("_").replace({"nan": np.nan, "": np.nan})
    return df

def replace_values(df, column_name, value_mapping):
    """Replace specified values in a column."""
    df[column_name] = df[column_name].replace(value_mapping)
    return df

def process_data(data_path):
    """
    Processes and cleans the dataset.
    Steps include dropping unnecessary columns, handling missing data, 
    transforming categorical features, and encoding.
    """
    logging.info("Loading data from: %s", data_path)
    df = pd.read_csv(data_path, dtype={"Monthly_Balance": "str"})

    # Drop unnecessary columns
    df = df.drop(columns=["ID", "Customer_ID", "Name", "SSN", "Month"])

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Strip excess characters from columns (other than Occupation)
    df = strip_values(df, df.columns)

    # Handle missing values for specific columns
    df = replace_values(df, "Occupation", {"_______": "Other"})
    df = replace_values(df, "Num_Bank_Accounts", {-1: np.nan})
    df["Num_of_Loan"] = np.where(df["Num_of_Loan"].astype(int) < 0, 0, df['Num_of_Loan'])
    df["Delay_from_due_date"] = np.where(df["Delay_from_due_date"].astype(int) < 0, np.nan, df["Delay_from_due_date"])
    df = replace_values(df, "Credit_Mix", {"" : np.nan})
    df = replace_values(df, "Payment_Behaviour", {"!@9#%8" : np.nan})
    df = replace_values(df, "Payment_of_Min_Amount", {"NM" : np.nan})
    df = replace_values(df, "Changed_Credit_Limit", {"_" : np.nan})
    df = replace_values(df, "Amount_invested_monthly", {"__10000__" : 10000.00})

    df.replace("nan", np.nan, inplace=True)

    # Drop any rows with NA values
    df.dropna(inplace=True)

    # Convert age in years to age in months
    df["Age"] = df["Age"].astype(int) * 12

    # Convert Credit_History_Age from string format to months
    df["Credit_History_Age"] = df["Credit_History_Age"].str.replace(" and", "").str.split().apply(lambda s: int(s[0]) * 12 + int(s[2]))

    # One-hot encode occupation types
    occupation_df = pd.get_dummies(df["Occupation"], prefix="Occ", drop_first=True)
    df = pd.concat([df, occupation_df], axis=1)
    df.drop(columns=["Occupation"], inplace=True)

    # One-hot encode loan types using MultiLabelBinarizer
    df["Type_of_Loan"] = df["Type_of_Loan"].apply(lambda s: list(set([loan.strip() for loan in s.replace("and", "").split(",") if loan.strip() != ""])))
    encoded_loan = mlb.fit_transform(df["Type_of_Loan"])
    loan_df = pd.DataFrame(encoded_loan, columns=mlb.classes_, index=df.index)
    df = pd.concat([df, loan_df], axis=1)
    df.drop(columns=["Type_of_Loan"], inplace=True)

    # Ordinal encode Credit_Mix and Payment_of_Min_Amount
    credit_mix_mapping = {"Bad": 0, "Standard": 1, "Good": 2}
    df["Credit_Mix"] = df["Credit_Mix"].map(credit_mix_mapping)
    yes_no_mapping = {"No": 0, "Yes": 1}
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].map(yes_no_mapping)

    # One-hot encode Payment_Behaviour
    def parse_payment_behaviour(s):
        parts = s.split('_')
        return parts[0], parts[2]

    df[['Spend_Level', 'Payment_Size']] = df['Payment_Behaviour'].apply(lambda s: pd.Series(parse_payment_behaviour(s)))
    spend_mapping = {"Low": 0, "High": 1}
    size_mapping = {"Small": 0, "Medium": 1, "Large": 2}
    df["Spend_Level"] = df["Spend_Level"].map(spend_mapping)
    df["Payment_Size"] = df["Payment_Size"].map(size_mapping)
    df.drop(columns=["Payment_Behaviour"], inplace=True)

    # If train data, encode Credit_Score
    if "Credit_Score" in df.columns:
        credit_score_mapping = {"Poor": 0, "Standard": 1, "Good": 2}
        df["Credit_Score"] = df["Credit_Score"].map(credit_score_mapping)

    # Convert all values to float
    df = df.astype(float)

    logging.info("Data processed successfully.")
    return df

def standardize_data(df):
    """
    Standardize numerical columns and ensure they are positive.
    """
    logging.info("Standardizing data...")
    standardized_data = df.copy()

    num_columns = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", 
                   "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", 
                   "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age", 
                   "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
                   "Changed_Credit_Limit", "Num_of_Delayed_Payment"]
    
    # Standardize the numerical columns
    standardized_data[num_columns] = scaler.fit_transform(df[num_columns])

    # Shift data up by min value so that values are positive
    for col in num_columns:
        min_value = standardized_data[col].min()
        if min_value < 0:
            shift_value = abs(min_value) + 1
            standardized_data[col] += shift_value
    
    logging.info("Data standardized successfully.")
    return standardized_data

def process_input_data(df):
    """
    Processes and cleans input data from a DataFrame.
    Similar to process_data but without file reading operations.
    """
    # Strip excess characters from columns (other than Occupation)
    df = strip_values(df, df.columns)

    # Handle missing values for specific columns
    df = replace_values(df, "Occupation", {"_______": "Other"})
    df = replace_values(df, "Num_Bank_Accounts", {-1: np.nan})
    df["Number_of_Loans"] = np.where(df["Number_of_Loans"].astype(int) < 0, 0, df['Number_of_Loans'])
    df["Delay_From_Due_Date"] = np.where(df["Delay_From_Due_Date"].astype(int) < 0, np.nan, df["Delay_From_Due_Date"])
    df = replace_values(df, "Credit_Mix", {"" : np.nan})
    df = replace_values(df, "Payment_of_Minimum_Amount", {"NM" : np.nan})
    df = replace_values(df, "Changed_Credit_Limit", {"_" : np.nan})
    df = replace_values(df, "Amount_Invested_Monthly", {"__10000__" : 10000.00})

    df.replace("nan", np.nan, inplace=True)

    # Convert age in years to age in months
    df["Age"] = df["Age"].astype(int) * 12

    # Convert Credit_History_Age from string format to months
    df["Credit_History_Age"] = df["Credit_History_Age"].str.replace(" and", "").str.split().apply(lambda s: int(s[0]) * 12 + int(s[2]))

    # One-hot encode occupation types
    occupation_df = pd.get_dummies(df["Occupation"], prefix="Occ", drop_first=True)
    df = pd.concat([df, occupation_df], axis=1)
    df.drop(columns=["Occupation"], inplace=True)

    # One-hot encode loan types using MultiLabelBinarizer
    df["Loan_Type"] = df["Loan_Type"].apply(lambda s: list(set([loan.strip() for loan in s.replace("and", "").split(",") if loan.strip() != ""])))
    encoded_loan = mlb.fit_transform(df["Loan_Type"])
    loan_df = pd.DataFrame(encoded_loan, columns=mlb.classes_, index=df.index)
    df = pd.concat([df, loan_df], axis=1)
    df.drop(columns=["Loan_Type"], inplace=True)

    # Ordinal encode Credit_Mix and Payment_of_Min_Amount
    credit_mix_mapping = {"Bad": 0, "Standard": 1, "Good": 2}
    df["Credit_Mix"] = df["Credit_Mix"].map(credit_mix_mapping)
    yes_no_mapping = {"No": 0, "Yes": 1}
    df["Payment_of_Minimum_Amount"] = df["Payment_of_Minimum_Amount"].map(yes_no_mapping)

    # One-hot encode Payment_Behaviour
    def parse_payment_behaviour(s):
        parts = s.split('_')
        return parts[0], parts[2]

    df[['Spend_Level', 'Payment_Size']] = df['Payment_Behaviour'].apply(lambda s: pd.Series(parse_payment_behaviour(s)))
    spend_mapping = {"Low": 0, "High": 1}
    size_mapping = {"Small": 0, "Medium": 1, "Large": 2}
    df["Spend_Level"] = df["Spend_Level"].map(spend_mapping)
    df["Payment_Size"] = df["Payment_Size"].map(size_mapping)
    df.drop(columns=["Payment_Behaviour"], inplace=True)

    # Convert all values to float
    df = df.astype(float)

    df = standardize_data(df)

    return df

def main():    
    # Load and process train and test data
    df_train = process_data("/Users/evanshi/Desktop/Personal-Projects/AICreditScoring/data/raw/train.csv")
    df_test = process_data("/Users/evanshi/Desktop/Personal-Projects/AICreditScoring/data/raw/test.csv")

    # Standardize the processed data
    standardized_train = standardize_data(df_train)
    standardized_test = standardize_data(df_test)

    # Optionally save the processed data (uncomment these lines to save)
    standardized_train.to_csv(train_file_path, index=False)
    standardized_test.to_csv(test_file_path, index=False)


if __name__ == "__main__":
    main()
