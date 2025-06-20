import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
import logging
import os
import joblib

# retrieve file path from env
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

# Initialize preprocessors
mlb = MultiLabelBinarizer()
scaler = StandardScaler()
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)

def save_preprocessors():
    """Save the fitted preprocessors to disk."""
    models_dir = os.getenv("MODELS_DIR")
    if not models_dir:
        raise ValueError("MODELS_DIR environment variable is not set")
    
    joblib.dump(mlb, os.path.join(models_dir, 'multilabel_binarizer.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'standard_scaler.pkl'))
    joblib.dump(ohe, os.path.join(models_dir, 'occupation_ohe.pkl'))
    # Save feature columns
    joblib.dump(feature_columns, os.path.join(models_dir, 'feature_columns.pkl'))
    logging.info("Preprocessors and feature columns saved successfully")

def load_preprocessors():
    """Load the fitted preprocessors from disk."""
    models_dir = os.getenv("MODELS_DIR")
    if not models_dir:
        raise ValueError("MODELS_DIR environment variable is not set")
    
    mlb_path = os.path.join(models_dir, 'multilabel_binarizer.pkl')
    scaler_path = os.path.join(models_dir, 'standard_scaler.pkl')
    ohe_path = os.path.join(models_dir, 'occupation_ohe.pkl')
    feature_columns_path = os.path.join(models_dir, 'feature_columns.pkl')
    
    if not os.path.exists(mlb_path) or not os.path.exists(scaler_path) or not os.path.exists(ohe_path) or not os.path.exists(feature_columns_path):
        raise FileNotFoundError("Preprocessor files not found. Please run training first.")
    
    global mlb, scaler, ohe, feature_columns
    mlb = joblib.load(mlb_path)
    scaler = joblib.load(scaler_path)
    ohe = joblib.load(ohe_path)
    feature_columns = joblib.load(feature_columns_path)
    logging.info("Preprocessors and feature columns loaded successfully")

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

# Convert age in years to age in months
def convert_age(df):
    df["Age"] = df["Age"].astype(int) * 12
    return df

# Convert Credit_History_Age from string format to months
def convert_credit_history_age(age_val):
    # If it's already an integer or float, return as int
    if isinstance(age_val, (int, float)):
        return int(age_val)
    try:
        # Try to parse string like "1 Year and 11 Months"
        parts = age_val.replace(" and ", " ").split()
        years = int(parts[0])
        months = int(parts[2])
        return years * 12 + months
    except Exception:
        # If parsing fails, try to convert directly to int
        try:
            return int(age_val)
        except Exception:
            return 0

# one hot encode occupations with sklearn ohe
def one_hot_encode_occupations(df, fit=False):
    global ohe
    if fit:
        occupation_encoded = ohe.fit_transform(df[["Occupation"]])
    else:
        occupation_encoded = ohe.transform(df[["Occupation"]])
    occupation_df = pd.DataFrame(occupation_encoded, columns=ohe.get_feature_names_out(["Occupation"]), index=df.index)
    df = pd.concat([df.drop("Occupation", axis=1), occupation_df], axis=1)
    return df

# One-hot encode loan types using MultiLabelBinarizer
def one_hot_encode_loan_types(df):
    df["Type_of_Loan"] = df["Type_of_Loan"].apply(lambda s: list(set([loan.strip() for loan in s.replace("and", "").split(",") if loan.strip() != ""])))
    encoded_loan = mlb.fit_transform(df["Type_of_Loan"])
    loan_df = pd.DataFrame(encoded_loan, columns=mlb.classes_, index=df.index)
    df = pd.concat([df, loan_df], axis=1)
    df.drop(columns=["Type_of_Loan"], inplace=True)
    return df

# ordinal encode Credit_mix
def ordinal_encode_credit_mix(df):
    credit_mix_mapping = {"Bad": 0, "Standard": 1, "Good": 2}
    df["Credit_Mix"] = df["Credit_Mix"].map(credit_mix_mapping)
    return df

# ordinal encode Payment_of_Min_Amount
def ordinal_encode_payement_of_min_amount(df):
    yes_no_mapping = {"No": 0, "Yes": 1}
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].map(yes_no_mapping)
    return df


# parse Payment_Behaviour
def parse_payment_behaviour(df):

    def split_payment_behaviour(s):
        parts = s.split('_')
        return parts[0], parts[2]

    df[['Spend_Level', 'Payment_Size']] = df['Payment_Behaviour'].apply(lambda s: pd.Series(split_payment_behaviour(s)))
    df.drop(columns=["Payment_Behaviour"], inplace=True)
    return df


# one-hot encode Spend_Level
def one_hot_encode_spend_level(df):
    spend_mapping = {"Low": 0, "High": 1}
    df["Spend_Level"] = df["Spend_Level"].map(spend_mapping)
    return df

# one-hot encode Payment_Size
def one_hot_encode_payment_size(df):
    size_mapping = {"Small": 0, "Medium": 1, "Large": 2}
    df["Payment_Size"] = df["Payment_Size"].map(size_mapping)
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

    # convert formats
    df = convert_age(df)
    df["Credit_History_Age"] = df["Credit_History_Age"].apply(convert_credit_history_age)

    # encode categorical features
    df = one_hot_encode_occupations(df, fit=True)
    df = one_hot_encode_loan_types(df)
    df = ordinal_encode_credit_mix(df)
    df = ordinal_encode_payement_of_min_amount(df)

    # parse payment_behaviour
    df = parse_payment_behaviour(df)

    # encode spend level and payment size
    df = one_hot_encode_spend_level(df)
    df = one_hot_encode_payment_size(df)

    # If train data, encode Credit_Score
    if "Credit_Score" in df.columns:
        credit_score_mapping = {"Poor": 0, "Standard": 1, "Good": 2}
        df["Credit_Score"] = df["Credit_Score"].map(credit_score_mapping)

    # Convert all values to float
    df = df.astype(float)

    # After all processing, save the feature columns
    global feature_columns
    feature_columns = df.columns.tolist()

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
    
    # Standardize the numerical columns using the loaded scaler
    standardized_data[num_columns] = scaler.fit_transform(df[num_columns])
    
    # Shift data up by min value so that values are positive
    for col in num_columns:
        min_value = standardized_data[col].values.min()
        if min_value < 0:
            shift_value = abs(min_value) + 1
            standardized_data[col] += shift_value
    
    logging.info("Data standardized successfully.")
    return standardized_data

def process_input_data(df):
    # Load preprocessors if not already loaded
    try:
        load_preprocessors()
    except FileNotFoundError:
        logging.warning("Preprocessor files not found. Using default encoding.")
    
    # Log the input columns
    print("Input DataFrame columns:", df.columns.tolist())
    
    # Strip excess characters from columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    # Handle missing values for specific columns
    df["Num_Bank_Accounts"] = df["Num_Bank_Accounts"].fillna(0)
    df["Num_Credit_Card"] = df["Num_Credit_Card"].fillna(0)
    df["Interest_Rate"] = df["Interest_Rate"].fillna(0)
    df["Num_of_Loan"] = df["Num_of_Loan"].fillna(0)
    df["Delay_from_due_date"] = df["Delay_from_due_date"].fillna(0)
    df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].fillna(0)
    df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].fillna(0)
    df["Outstanding_Debt"] = df["Outstanding_Debt"].fillna(0)
    df["Credit_Utilization_Ratio"] = df["Credit_Utilization_Ratio"].fillna(0)
    df["Credit_History_Age"] = df["Credit_History_Age"].fillna("0 Years and 0 Months")
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].fillna("No")
    df["Total_EMI_per_month"] = df["Total_EMI_per_month"].fillna(0)
    df["Amount_invested_monthly"] = df["Amount_invested_monthly"].fillna(0)
    df["Monthly_Balance"] = df["Monthly_Balance"].fillna(0)
    
    # Convert age from years to months
    df["Age"] = df["Age"] * 12
    
    # Convert Credit_History_Age from string to months
    df["Credit_History_Age"] = df["Credit_History_Age"].apply(convert_credit_history_age)
    
    # One-hot encode Occupation using loaded OneHotEncoder (fit=False)
    df = one_hot_encode_occupations(df, fit=False)
    
    # One-hot encode Loan_Type using the loaded MultiLabelBinarizer
    df["Type_of_Loan"] = df["Loan_Type"].apply(lambda s: list(set([loan.strip() for loan in s.replace("and", "").split(",") if loan.strip() != ""])))
    encoded_loan = mlb.transform(df["Type_of_Loan"]) 
    loan_df = pd.DataFrame(encoded_loan, columns=mlb.classes_, index=df.index)
    df = pd.concat([df, loan_df], axis=1)
    df = df.drop(["Loan_Type", "Type_of_Loan"], axis=1)
    
    # Ordinal encode Credit_Mix
    credit_mix_map = {"Bad": 0, "Standard": 1, "Good": 2}
    df["Credit_Mix"] = df["Credit_Mix"].map(credit_mix_map)
    
    # Ordinal encode Payment_of_Minimum_Amount
    payment_map = {"No": 0, "Yes": 1}
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].map(payment_map)
    
    # Map Spend_Level and Payment_Size directly
    spend_mapping = {"Low": 0, "High": 1}
    size_mapping = {"Small": 0, "Medium": 1, "Large": 2}
    df["Spend_Level"] = df["Spend_Level"].map(spend_mapping)
    df["Payment_Size"] = df["Payment_Size"].map(size_mapping)
    
    # Convert all values to float
    for col in df.columns:
        if col not in ["Customer_ID", "Name", "SSN"]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Apply the same shift values as used during training
    df["Age"] = df["Age"] + 240 
    df["Annual_Income"] = df["Annual_Income"] + 10000  
    df["Monthly_Inhand_Salary"] = df["Monthly_Inhand_Salary"] + 1000 
    df["Num_Bank_Accounts"] = df["Num_Bank_Accounts"] + 1 
    df["Num_Credit_Card"] = df["Num_Credit_Card"] + 1  
    df["Interest_Rate"] = df["Interest_Rate"] + 5  
    df["Num_of_Loan"] = df["Num_of_Loan"] + 1 
    df["Delay_from_due_date"] = df["Delay_from_due_date"] + 5  
    df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"] + 1  
    df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"] + 1000  
    df["Outstanding_Debt"] = df["Outstanding_Debt"] + 1000  
    df["Credit_Utilization_Ratio"] = df["Credit_Utilization_Ratio"] + 10 
    df["Credit_History_Age"] = df["Credit_History_Age"] + 12  
    df["Total_EMI_per_month"] = df["Total_EMI_per_month"] + 100  
    df["Amount_invested_monthly"] = df["Amount_invested_monthly"] + 100  
    df["Monthly_Balance"] = df["Monthly_Balance"] + 1000 
    
    # Add missing columns and reorder to match training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Final columns: {df.columns.tolist()}")
    
    return df

def main():    
    # Load and process train and test data
    df_train = process_data(DATA_DIR + "/raw/train.csv")
    df_test = process_data(DATA_DIR + "/raw/test.csv")

    # Standardize the processed data
    standardized_train = standardize_data(df_train)
    standardized_test = standardize_data(df_test)

    # Save the fitted preprocessors
    save_preprocessors()

    # Optionally save the processed data (uncomment these lines to save)
    standardized_train.to_csv(DATA_DIR + "/processed/processed_train", index=False)
    standardized_test.to_csv(DATA_DIR + "/processed/processed_test", index=False)


if __name__ == "__main__":
    main()
