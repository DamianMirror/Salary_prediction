# Salary Prediction App

A simple FastAPI web application that predicts whether a person's annual income exceeds $50K based on census data.

## Features

- CatBoost machine learning model with 87% accuracy
- Simple one-page web interface
- Real-time predictions with probability scores
- RESTful API endpoint

## Setup

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Train and save the model:
```bash
python train_and_save_model.py
```

This will create the following files in the `backend/` directory:
- `model.pkl` - Trained CatBoost model
- `scaler.pkl` - StandardScaler for numeric features
- `label_encoder.pkl` - LabelEncoder for gender
- `feature_columns.pkl` - Feature column names

3. Run the FastAPI server:
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload
```

4. Open your browser and go to:
```
http://localhost:8000
```

## API Endpoints

### GET /
Returns the web interface

### POST /predict
Predicts salary category based on input features

**Request Body:**
```json
{
  "age": 35,
  "workclass": "Private",
  "education": "Bachelors",
  "educational_num": 13,
  "marital_status": "Married-civ-spouse",
  "occupation": "Prof-specialty",
  "relationship": "Husband",
  "race": "White",
  "gender": "Male",
  "capital_gain": 0,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}
```

**Note**: In the web interface, users only select the education level (e.g., "Bachelors"), and the `educational_num` is automatically encoded from the education selection.

**Response:**
```json
{
  "prediction": "Income > $50K",
  "probability": 0.7234,
  "confidence": "72.34%"
}
```

### GET /health
Health check endpoint

## Model Details

- **Algorithm**: CatBoost Classifier
- **Test Accuracy**: ~87%
- **Test F1 Score**: ~0.73
- **Test ROC-AUC**: ~0.87
- **Decision Threshold**: 0.46

## Input Features

The model requires 13 input features:

1. **age**: Age in years (17-90)
2. **workclass**: Type of employment
3. **education**: Highest education level (automatically encoded to educational_num in the web interface)
4. **educational_num**: Years of education (1-16, derived from education)
5. **marital_status**: Marital status
6. **occupation**: Type of occupation
7. **relationship**: Family relationship
8. **race**: Race
9. **gender**: Gender (Male/Female)
10. **capital_gain**: Capital gains
11. **capital_loss**: Capital losses
12. **hours_per_week**: Hours worked per week
13. **native_country**: Country of origin

**Note**: The `fnlwgt` (final weight) feature has been removed from the model as it represents census sampling weights and is not meaningful for individual predictions.

## Project Structure

```
Salary_prediction/
├── backend/
│   ├── app.py                    # FastAPI backend
│   ├── train_and_save_model.py  # Script to train and save model
│   ├── requirements.txt          # Python dependencies
│   ├── model.pkl                 # Saved model (generated)
│   ├── scaler.pkl                # Saved scaler (generated)
│   ├── label_encoder.pkl         # Saved encoder (generated)
│   └── feature_columns.pkl       # Feature columns (generated)
├── frontend/
│   └── index.html                # Frontend web interface
├── notebooks/
│   └── project.ipynb             # Jupyter notebook with analysis
├── train.csv                     # Training data
└── README.md                     # This file
```

## Technologies Used

- **Backend**: FastAPI, Uvicorn
- **ML**: CatBoost, Scikit-learn, Imbalanced-learn
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Data Processing**: Pandas, NumPy
