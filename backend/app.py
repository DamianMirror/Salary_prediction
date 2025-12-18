from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import Literal

app = FastAPI(title="Salary Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model and preprocessing objects...")
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run train_and_save_model.py first!")

class SalaryInput(BaseModel):
    age: int
    workclass: Literal[
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
    ]
    education: Literal[
        'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
        'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
        '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'
    ]
    educational_num: int
    marital_status: Literal[
        'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
        'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
    ]
    occupation: Literal[
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ]
    relationship: Literal[
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
    ]
    race: Literal['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    gender: Literal['Male', 'Female']
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.get("/")
async def read_root():
    return FileResponse('../frontend/index.html')

@app.post("/predict")
async def predict(input_data: SalaryInput):
    try:
        input_dict = {
            'age': input_data.age,
            'educational-num': input_data.educational_num,
            'capital-gain': input_data.capital_gain,
            'capital-loss': input_data.capital_loss,
            'hours-per-week': input_data.hours_per_week,
            'gender': input_data.gender,
            'workclass': input_data.workclass,
            'marital-status': input_data.marital_status,
            'occupation': input_data.occupation,
            'education': input_data.education,
            'race': input_data.race,
            'relationship': input_data.relationship,
            'native-country': input_data.native_country
        }

        df = pd.DataFrame([input_dict])

        df['gender'] = label_encoder.transform(df['gender'])

        df = pd.get_dummies(df, columns=[
            'workclass', 'education', 'marital-status',
            'occupation', 'race', 'relationship', 'native-country'
        ], drop_first=True)

        num_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        df[num_cols] = scaler.transform(df[num_cols])

        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]

        proba = model.predict_proba(df)[0, 1]
        prediction = int(proba >= 0.46)

        result = {
            "prediction": "Income > $50K" if prediction == 1 else "Income ≤ $50K",
            "probability": float(proba),
            "confidence": f"{float(proba) * 100:.2f}%" if prediction == 1 else f"{(1 - float(proba)) * 100:.2f}%"
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
