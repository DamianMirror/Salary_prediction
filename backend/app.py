from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import Literal
import io

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

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")

        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        required_columns = [
            'age', 'workclass', 'education', 'educational-num', 'marital-status',
            'occupation', 'relationship', 'race', 'gender', 'capital-gain',
            'capital-loss', 'hours-per-week', 'native-country'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )

        # Check for missing values in required columns
        df = df[required_columns].copy()
        if df.isnull().any().any():
            # Fill missing values with defaults
            df['age'] = df['age'].fillna(30)
            df['educational-num'] = df['educational-num'].fillna(9)
            df['capital-gain'] = df['capital-gain'].fillna(0)
            df['capital-loss'] = df['capital-loss'].fillna(0)
            df['hours-per-week'] = df['hours-per-week'].fillna(40)
            df['gender'] = df['gender'].fillna('Male')
            df['workclass'] = df['workclass'].fillna('Private')
            df['marital-status'] = df['marital-status'].fillna('Never-married')
            df['occupation'] = df['occupation'].fillna('Other-service')
            df['education'] = df['education'].fillna('HS-grad')
            df['race'] = df['race'].fillna('White')
            df['relationship'] = df['relationship'].fillna('Not-in-family')
            df['native-country'] = df['native-country'].fillna('United-States')

        original_df = df.copy()

        df['gender'] = label_encoder.transform(df['gender'])

        df_encoded = pd.get_dummies(df, columns=[
            'workclass', 'education', 'marital-status',
            'occupation', 'race', 'relationship', 'native-country'
        ], drop_first=True)

        num_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])

        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[feature_columns]

        probas = model.predict_proba(df_encoded)[:, 1]
        predictions = (probas >= 0.46).astype(int)

        results = []
        for idx, (pred, proba) in enumerate(zip(predictions, probas)):
            result = original_df.iloc[idx].to_dict()

            # Replace NaN/Inf values with None for JSON serialization
            for key, value in result.items():
                if isinstance(value, (float, np.floating)):
                    if np.isnan(value) or np.isinf(value):
                        result[key] = None
                    else:
                        result[key] = float(value)

            # Ensure probability is valid
            proba_value = float(proba)
            if np.isnan(proba_value) or np.isinf(proba_value):
                proba_value = 0.5  # default to 50% if invalid

            result['prediction'] = "Income > $50K" if pred == 1 else "Income ≤ $50K"
            result['probability'] = proba_value
            result['confidence'] = f"{proba_value * 100:.2f}%" if pred == 1 else f"{(1 - proba_value) * 100:.2f}%"
            results.append(result)

        return {"predictions": results, "total_count": len(results)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV prediction error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
