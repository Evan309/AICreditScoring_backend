from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import Optional
import os


from data.data_cleaning import process_input_data, standardize_data
from models.NeuralNetwork import Layer_Dense, Activation_ReLU, Activation_Softmax

# Initialize FastAPI app
app = FastAPI(title="Credit Score Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input data model
class CreditScoreInput(BaseModel):
    Age: Optional[float] = None
    Annual_Income: Optional[float] = None
    Monthly_Inhand_Salary: Optional[float] = None
    Occupation: Optional[str] = None
    Num_Bank_Accounts: Optional[float] = None
    Num_Credit_Card: Optional[float] = None
    Monthly_Balance: Optional[float] = None
    Changed_Credit_Limit: Optional[float] = None
    Credit_History_Age: Optional[str] = None
    Outstanding_Debt: Optional[float] = None
    Number_of_Delayed_Payment: Optional[float] = None
    Credit_Mix: Optional[str] = None
    Credit_Utilization_Ratio: Optional[float] = None
    Delay_From_Due_Date: Optional[float] = None
    Number_of_Loans: Optional[float] = None
    Interest_Rate: Optional[float] = None
    Total_EMI_Per_Month: Optional[float] = None
    Loan_Type: Optional[str] = None
    Payment_of_Minimum_Amount: Optional[str] = None
    Spend_Level: Optional[str] = None
    Payment_Size: Optional[str] = None
    Amount_Invested_Monthly: Optional[float] = None

class CreditScoreModel:
    def __init__(self):
        self.dense1 = None
        self.activation1 = Activation_ReLU()
        self.dense2 = None
        self.activation2 = Activation_ReLU()
        self.dense3 = None
        self.activation3 = Activation_ReLU()
        self.dense4 = None
        self.activation4 = Activation_Softmax()
        
    def load(self, model_path):
        # Load the model parameters
        model_data = np.load(model_path)
        
        # Initialize layers with correct dimensions
        self.dense1 = Layer_Dense(model_data['input_size'][0], model_data['hidden_size'][0])
        self.dense2 = Layer_Dense(model_data['hidden_size'][0], model_data['hidden_size'][0])
        self.dense3 = Layer_Dense(model_data['hidden_size'][0], model_data['hidden_size'][0])
        self.dense4 = Layer_Dense(model_data['hidden_size'][0], model_data['output_size'][0])
        
        # Load weights and biases
        self.dense1.weights = model_data['dense1_weights']
        self.dense1.biases = model_data['dense1_biases']
        self.dense2.weights = model_data['dense2_weights']
        self.dense2.biases = model_data['dense2_biases']
        self.dense3.weights = model_data['dense3_weights']
        self.dense3.biases = model_data['dense3_biases']
        self.dense4.weights = model_data['dense4_weights']
        self.dense4.biases = model_data['dense4_biases']
    
    def predict(self, X):
        # Forward pass
        self.dense1.forward(X)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)
        self.dense3.forward(self.activation2.output)
        self.activation3.forward(self.dense3.output)
        self.dense4.forward(self.activation3.output)
        self.activation4.forward(self.dense4.output)
        
        # Get prediction
        prediction = np.argmax(self.activation4.output, axis=1)
        confidence = np.max(self.activation4.output, axis=1)
        
        return prediction[0], confidence[0]

# Initialize model
model = CreditScoreModel()

# Load model
def load_model():
    try:
        models_dir = os.getenv("MODELS_DIR")
        model_path = os.path.join(models_dir, '/credit_scoring_model.npz')
        model.load(model_path)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Preprocess input data
def preprocess_input(data: CreditScoreInput):
    # Convert input data to DataFrame
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])
    
    # Process the data using the input processing pipeline
    processed_df = process_input_data(df)
    
    # Standardize the data
    standardized_df = standardize_data(processed_df)
    
    # Convert to numpy array for prediction
    features = standardized_df.values
    
    return features

@app.post("/predict")
async def predict_credit_score(data: CreditScoreInput):
    try:
        # Load model if not already loaded
        if model.dense1 is None:
            load_model()
        
        # Preprocess input using the data cleaning pipeline
        features = preprocess_input(data)
        
        # Make prediction
        prediction, confidence = model.predict(features)
        
        # Map prediction to credit score category
        credit_score_map = {0: "Poor", 1: "Standard", 2: "Good"}
        credit_score = credit_score_map[prediction]
        
        return {
            "score": credit_score,
            "confidence": f"{confidence * 100:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Credit Score Prediction API is running"}
