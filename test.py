import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle

# Genetic marker columns
GENETIC_MARKER_COLUMNS = [
    'rs10007052', 'rs8192288', 'rs20541', 'rs12922394',
    'rs2910164', 'rs161976', 'rs473892', 'rs159497', 'rs9296092'
]


def validate_and_clean_data(data, feature_columns):
    """Validate and clean the dataset."""
    for col in feature_columns:
        if col in data.columns:
            # Convert non-numeric values to NaN for numeric columns
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Log invalid rows
    invalid_rows = data[data.isnull().any(axis=1)]
    if not invalid_rows.empty:
        print(f"Invalid rows detected: {len(invalid_rows)} rows logged.")
        # Save invalid rows for debugging
        invalid_rows.to_csv("invalid_rows_log.csv", index=False)

    return data


def load_and_prepare_data(file_path, asthma_file_path):
    """Load and prepare datasets for training."""
    try:
        data = pd.read_csv(file_path)
        asthma_data = pd.read_csv(asthma_file_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        raise

    # Prepare asthma_data
    asthma_data = asthma_data.rename(columns={
        'Patient_ID': 'uid',
        'Age': 'age',
        'Gender': 'sex',
        'Smoking_Status': 'smoke'
    })
    asthma_data['sex'] = asthma_data['sex'].map({'Male': 1, 'Female': 2})
    asthma_data['smoke'] = asthma_data['smoke'].map({'Yes': 1, 'No': 0})

    for col in GENETIC_MARKER_COLUMNS:
        asthma_data[col] = pd.NA

    asthma_data['risk_level'] = 'High Risk'
    asthma_data.drop(columns=['Medication', 'Peak_Flow'], inplace=True, errors='ignore')
    combined_data = pd.concat([data, asthma_data], ignore_index=True)

    # Ensure genetic_marker_columns exist in combined_data
    for col in GENETIC_MARKER_COLUMNS:
        if col not in combined_data.columns:
            combined_data[col] = np.nan
    combined_data = validate_and_clean_data(combined_data, ['age', 'bmi', 'smoke'] + GENETIC_MARKER_COLUMNS)

    return combined_data


def assign_risk_level(row):
    """Assign risk level based on the row attributes."""
    if row['label'] == 0:
        if row['smoke'] == 0 and row['age'] < 45 and row['bmi'] < 25:
            return 'Low Risk'
        elif row['age'] < 65 and row['bmi'] < 30:
            return 'Moderate Risk'
    elif row['label'] == 1:
        if row['age'] >= 65 and row['bmi'] >= 25:
            return 'High Risk'
        elif row['age'] >= 65 and row['bmi'] >= 30 and row['smoke'] == 1:
            return 'Severe Risk'
    return 'Moderate Risk'


def preprocess_data(combined_data):
    """Preprocess the data: impute missing values and encode labels."""
    # Assign risk levels
    combined_data['risk_level'] = combined_data.apply(assign_risk_level, axis=1)
    # Prepare features and target
    features = combined_data[['age', 'bmi', 'smoke'] + GENETIC_MARKER_COLUMNS]
    target = combined_data['risk_level']

    missing_data = features.isnull().sum()
    if missing_data.any():
        print("Missing or invalid values detected in features:\n", missing_data)

    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)

    # Impute missing values in genetic marker columns
    imputer = SimpleImputer(strategy="median")
    features[GENETIC_MARKER_COLUMNS] = imputer.fit_transform(features[GENETIC_MARKER_COLUMNS])

    return features, target_encoded, label_encoder, imputer


def train_model(features, target_encoded):
    """Train the XGBoost model."""
    X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)
    model = XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(report)
    print("Confusion Matrix:\n", conf_matrix)

    return model, X_test, y_test


if __name__ == "__main__":
    FILE_PATH = "C:/Users/shiva/Downloads/Finalalldata.csv"
    ASTHMA_FILE_PATH = "C:/Users/shiva/Downloads/asthma_dataset.csv"

    try:
        # Load and prepare data
        combined_data = load_and_prepare_data(FILE_PATH, ASTHMA_FILE_PATH)

        # Preprocess data
        features, target_encoded, label_encoder, imputer = preprocess_data(combined_data)

        # Train model
        model, X_test, y_test = train_model(features, target_encoded)

        # SHAP Integration
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Save model components for reuse in testing
        with open("model_components.pkl", "wb") as f:
            pickle.dump((model, label_encoder, imputer), f)
    except Exception as e:
        print(f"An error occurred: {e}")
