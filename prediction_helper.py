# codebasics ML course: codebasics.io, all rights reserved

import pandas as pd
import joblib

# Corrected file paths (use raw strings to avoid escape sequence issues)
model_young = joblib.load(r"ReadyModels\model_young.joblib")
model_rest = joblib.load(r"ReadyModels\model_rest.joblib")
scaler_young = joblib.load(r"ReadyModels\scaler_young.joblib")
scaler_rest = joblib.load(r"ReadyModels\scaler_rest.joblib")


def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = medical_history.lower().split(" & ")

    total_risk_score = sum(risk_scores.get(disease.strip(), 0) for disease in diseases)

    max_score = 14  # 8 (heart disease) + 6 (diabetes or high bp)
    min_score = 0

    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score


def preprocess_input(input_dict):
    # Define expected columns EXACTLY as in training
    expected_columns = [
        'age', 'number_of_dependants', 'income_level', 'income_lakhs', 'insurance_plan',
        'normalized_risk_score', 'gender_Male', 'region_Northwest', 'region_Southeast',
        'region_Southwest', 'marital_status_Unmarried', 'bmi_category_Obesity',
        'bmi_category_Overweight', 'bmi_category_Underweight',
        'smoking_status_Occasional', 'smoking_status_Regular',
        'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    # Initialize dataframe with zeros
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Manual encoding
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
            elif value == 'Freelancer':
                # Leave both as 0
                pass
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        # ⚠️ Removed genetical_risk since it wasn’t in training features

    # Calculate normalized risk
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])

    # Handle scaling
    df = handle_scaling(input_dict['Age'], df)

    return df


def handle_scaling(age, df):
    # Pick the right scaler
    scaler_object = scaler_young if age <= 25 else scaler_rest

    # Case A: If scaler_object is dict with metadata
    if isinstance(scaler_object, dict):
        scaler = scaler_object.get("scaler")
        cols_to_scale = scaler_object.get("cols_to_scale", [])
    else:
        # Case B: If it's just the scaler
        scaler = scaler_object
        # Assume we always want to scale numeric columns only
        cols_to_scale = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Ensure it's always list
    if isinstance(cols_to_scale, str):
        cols_to_scale = [cols_to_scale]

    # Debug
    print("Scaler type:", "young" if age <= 25 else "rest")
    print("Cols to scale:", cols_to_scale)

    # Scale only valid columns (intersection)
    valid_cols = [col for col in cols_to_scale if col in df.columns]
    if valid_cols:
        df[valid_cols] = scaler.transform(df[valid_cols])

    return df



def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])
