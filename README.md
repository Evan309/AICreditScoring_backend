# Credit Scoring Backend (FastAPI)

## Overview
This backend powers a modern credit scoring application, providing robust, production-ready APIs for real-time credit risk assessment. Built with FastAPI, it leverages advanced data cleaning, feature engineering, and a custom neural network model to deliver accurate credit score predictions. The architecture is modular, scalable, and designed for seamless integration with any frontend.

## Features
- **RESTful API** for credit score prediction
- **Advanced data preprocessing**: Handles missing values, categorical encoding, and feature standardization
- **Custom neural network inference**: Loads and runs a trained model for real-time predictions
- **Robust error handling** and logging for production reliability
- **Environment-based configuration** for flexible deployment
- **Open source (MIT License)**

## Tech Stack
- **Python 3.10+**
- **FastAPI** (API framework)
- **Pandas, NumPy** (data processing)
- **scikit-learn** (preprocessing)
- **Joblib** (model serialization)
- **Uvicorn** (ASGI server)
- **Pydantic** (data validation)
- **dotenv** (environment management)

## Custom Neural Network — Coded from Scratch
One of the standout features of this backend is the **fully custom Artificial Neural Network (ANN)**, implemented entirely from scratch using only **NumPy**—with no external machine learning libraries. This demonstrates a deep understanding of neural network theory and low-level ML engineering.

**Key Details:**
- **Architecture:** Multi-layer perceptron with several dense (fully connected) layers, ReLU and Softmax activations.
- **Training:** Forward and backward propagation, categorical cross-entropy loss, and Adam optimizer—all implemented manually.
- **Inference:** The trained model weights are saved as `.npz` files and loaded for real-time predictions in the API.
- **No external ML libraries:** All neural network logic (layers, activations, loss, optimizer) is hand-coded using NumPy arrays and operations.

This approach ensures full transparency, flexibility, and a strong demonstration of core ML and software engineering skills.

## API Endpoints
- `POST /predict` — Predicts a user's credit score category (`Poor`, `Standard`, `Good`) and returns a confidence score. Accepts a JSON payload with user financial and demographic features.
- `GET /` — Health check endpoint.

## Data Pipeline
- **Input validation**: Uses Pydantic models to ensure data integrity
- **Data cleaning**: Handles missing values, outliers, and inconsistent formats
- **Feature engineering**: One-hot/ordinal encoding, custom parsing (e.g., credit history age), and engineered features
- **Standardization**: Applies the same scaling and shifting as during model training for consistency

## Model Integration
- Loads a custom neural network from `.npz` weights
- Reconstructs architecture and performs forward inference
- Ensures all preprocessing matches the training pipeline for reliable predictions

## Getting Started
1. **Clone the repo:**
   ```bash
   git clone <your-repo-url>
   cd AICreditScoring_backend
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set environment variables:**
   - `MODELS_DIR`: Path to directory containing model and preprocessor files
   - `DATA_DIR`: Path to your data directory (for training/processing)
4. **Run the API server:**
   ```bash
   uvicorn src.api.main:app --reload
   ```

## Project Structure
```
AICreditScoring_backend/
├── src/
│   ├── api/
│   │   └── main.py         # FastAPI app and endpoints
│   ├── data/
│   │   └── data_cleaning.py # Data cleaning and preprocessing
│   └── models/
│       ├── NeuralNetwork.py # Model architecture (NumPy-only ANN)
│       └── ...              # Model weights, preprocessors
├── requirements.txt
├── LICENSE
└── README.md
```

## License
This project is licensed under the MIT License.

## About Me
I'm Evan, a passionate software engineer and machine learning enthusiast. This project demonstrates my ability to:
- Build production-grade APIs with FastAPI
- Engineer robust data pipelines for ML
- **Implement neural networks from scratch using only NumPy**
- Integrate custom ML models into real-world applications
- Write clean, maintainable, and well-documented code
- Debug and iterate quickly in real-world scenarios

I'm eager to bring these skills to a forward-thinking team. Let's connect! 