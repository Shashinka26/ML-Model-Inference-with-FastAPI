# Iris ML Model Inference API 🌸

A machine learning inference API built with **Python and FastAPI** for classifying Iris flowers using a trained machine learning model.

The API accepts the four standard Iris flower measurements and returns the predicted Iris species along with the model's confidence score when probability prediction is available.

---

## 🚀 Overview

This project demonstrates how a trained machine learning model can be served through a lightweight REST API using **FastAPI**.

The application loads a trained model from `model.pkl` when the API starts and provides endpoints for:

- API health checking
- Model information
- Iris flower prediction

The model classifies Iris flowers into:

- `setosa`
- `versicolor`
- `virginica`

---

## ✨ Features

- 🌸 Iris flower classification
- ⚡ FastAPI REST API
- 🤖 Pre-trained machine learning model
- 📊 Prediction confidence
- ✅ Pydantic input validation
- 🔍 Model information endpoint
- ❤️ API health check
- 💾 Serialized model loading with Joblib
- 🧪 Separate model training script

---

## 🛠️ Technology Stack

- Python
- FastAPI
- Pydantic
- NumPy
- Joblib
- Scikit-learn
- Uvicorn

---

## 📂 Project Structure

```text
ML-Model-Inference-with-FastAPI/
│
├── Data/
│
├── main.py
├── train_model.py
├── model.pkl
├── requirements.txt
├── .gitignore
└── README.md



File Description
File / Folder	Description
main.py	FastAPI application and prediction endpoints
train_model.py	Script used to train and save the ML model
model.pkl	Serialized trained model bundle
Data/	Project data
requirements.txt	Python dependencies
🧠 Machine Learning Model

The API uses four Iris flower measurements as input features:

Feature	Description
sepal_length	Sepal length in centimeters
sepal_width	Sepal width in centimeters
petal_length	Petal length in centimeters
petal_width	Petal width in centimeters

The model predicts one of the following classes:

setosa
versicolor
virginica

The trained model and additional metadata are loaded from:

model.pkl
🏗️ Architecture
Client
  │
  │ HTTP Request
  ▼
FastAPI
  │
  ▼
Pydantic Validation
  │
  ▼
NumPy Feature Array
  │
  ▼
Trained ML Model
  │
  ▼
Prediction + Confidence
  │
  ▼
JSON Response
🔌 API Endpoints
1. Health Check
GET /

Checks whether the API and trained model are available.

Example Response
{
  "status": "healthy",
  "message": "ML Model API is running"
}

If the model has not been loaded, the endpoint returns an error status and indicates that the model should be trained first.

2. Model Information
GET /model-info

Returns information about the loaded machine learning model.

Example Response
{
  "model_type": "SomeModel",
  "problem_type": "classification",
  "features": [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
  ],
  "classes": [
    "setosa",
    "versicolor",
    "virginica"
  ],
  "test_accuracy": 0.95
}

The exact model type and test accuracy are loaded from the trained model bundle.

3. Make a Prediction
POST /predict

Predicts the Iris flower species based on the supplied measurements.

Request Body
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
Response
{
  "prediction": "setosa",
  "confidence": 0.98
}

The confidence value is returned when the loaded model supports probability prediction.

🔄 Prediction Flow
Iris Measurements
       │
       ▼
POST /predict
       │
       ▼
Pydantic Validation
       │
       ▼
NumPy Array
       │
       ▼
ML Model
       │
       ▼
Predicted Class
       │
       ▼
Confidence Score
       │
       ▼
JSON Response
⚙️ Getting Started
Prerequisites

Make sure you have installed:

Python 3.x
pip
Git
1. Clone the Repository
git clone https://github.com/Shashinka26/ML-Model-Inference-with-FastAPI.git

Navigate into the project:

cd ML-Model-Inference-with-FastAPI
2. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Train the Model

If model.pkl needs to be generated again, run:

python train_model.py

This creates the trained model bundle used by the FastAPI application.

5. Start the API

Run:

uvicorn main:app --reload

The API will start on the local development server.

📚 Interactive API Documentation

FastAPI automatically provides interactive API documentation.

Once the server is running, open:

http://127.0.0.1:8000/docs

You can use Swagger UI to:

View available endpoints
Test /predict
Check /model-info
Test the health endpoint
Inspect request and response schemas
🧪 Example Prediction

Send a POST request to:

/predict

with:

{
  "sepal_length": 6.1,
  "sepal_width": 2.8,
  "petal_length": 4.7,
  "petal_width": 1.2
}

The API returns the predicted Iris class and, when supported by the model, its confidence score.

🔐 Error Handling

The API handles situations such as:

Model not being loaded
Invalid prediction input
Model prediction errors
Missing model.pkl

For example, if the model has not been trained or loaded, the API returns an appropriate error response indicating that the model needs to be created first.

🎯 Learning Objectives

This project demonstrates practical concepts including:

Machine learning model training
Model serialization
ML model inference
REST API development
FastAPI
Pydantic data validation
NumPy feature processing
Joblib model loading
Classification problems
Prediction probability
API documentation with Swagger
🔮 Future Improvements

Possible future improvements include:

Add automated tests
Add Docker support
Add CI/CD pipeline
Add model versioning
Add request logging
Add authentication
Add input range validation
Deploy the API to a cloud platform
Add a frontend interface for predictions
📌 Project Status

🚧 Learning / Development Project

This project was developed to explore machine learning model deployment and REST API development using FastAPI.

👨‍💻 Author

Chamidu Shashinka Rathnasiri

Software Developer

GitHub: https://github.com/Shashinka26
LinkedIn: https://www.linkedin.com/in/chamidu-shashinka-947709361
