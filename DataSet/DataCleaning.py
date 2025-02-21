import pandas as pd

df = pd.read_csv("/Users/evanshi/Desktop/Personal-Projects/AICreditScoring/DataSet/test.csv")

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

unique_jobs = df["occupation"].unique()
print(unique_jobs)
print(df.columns)

