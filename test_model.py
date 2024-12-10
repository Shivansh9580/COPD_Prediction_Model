import pickle
import pandas as pd
import shap
from sklearn.exceptions import NotFittedError


GENETIC_MARKER_COLUMNS = [
    'rs10007052', 'rs8192288', 'rs20541', 'rs12922394',
    'rs2910164', 'rs161976', 'rs473892', 'rs159497', 'rs9296092'
]


# Define test cases
def create_test_data():
    """Create test cases for prediction."""
    return pd.DataFrame([
        # Test case 1: Moderate Risk
        {
            'age': 50, 'bmi': 28.5, 'smoke': 1,
            'rs10007052': 1.2, 'rs8192288': 1.5, 'rs20541': 2.1,
            'rs12922394': 1.8, 'rs2910164': 1.9, 'rs161976': 2.5,
            'rs473892': 1.2, 'rs159497': 0.5, 'rs9296092': 1.1
        },
        # Test case 2: Low Risk
        {
            'age': 30, 'bmi': 22.0, 'smoke': 0,
            'rs10007052': 1.0, 'rs8192288': 1.0, 'rs20541': 1.0,
            'rs12922394': 1.0, 'rs2910164': 1.0, 'rs161976': 1.0,
            'rs473892': 1.0, 'rs159497': 1.0, 'rs9296092': 1.0
        },
        # Test case 3: High Risk
        {
            'age': 70, 'bmi': 28.0, 'smoke': 1,
            'rs10007052': 2.0, 'rs8192288': 2.0, 'rs20541': 2.0,
            'rs12922394': 2.0, 'rs2910164': 2.0, 'rs161976': 2.0,
            'rs473892': 2.0, 'rs159497': 2.0, 'rs9296092': 2.0
        },
        # Test case 4: Young non-smoker, low genetic marker values
        {
            'age': 25, 'bmi': 20.0, 'smoke': 0,
            'rs10007052': 0.8, 'rs8192288': 0.7, 'rs20541': 0.9,
            'rs12922394': 0.6, 'rs2910164': 0.7, 'rs161976': 0.8,
            'rs473892': 0.9, 'rs159497': 0.6, 'rs9296092': 0.7
        },
        # Test case 5: Middle-aged smoker, average genetic marker values
        {
            'age': 45, 'bmi': 25.0, 'smoke': 1,
            'rs10007052': 1.0, 'rs8192288': 1.0, 'rs20541': 1.0,
            'rs12922394': 1.0, 'rs2910164': 1.0, 'rs161976': 1.0,
            'rs473892': 1.0, 'rs159497': 1.0, 'rs9296092': 1.0
        },
        # Test case 6: Elderly obese smoker, high genetic marker values
        {
            'age': 70, 'bmi': 32.0, 'smoke': 1,
            'rs10007052': 2.0, 'rs8192288': 2.0, 'rs20541': 2.0,
            'rs12922394': 2.0, 'rs2910164': 2.0, 'rs161976': 2.0,
            'rs473892': 2.0, 'rs159497': 2.0, 'rs9296092': 2.0
        },
        # Test case 7: Middle-aged non-smoker, varied genetic marker values
        {
            'age': 50, 'bmi': 24.0, 'smoke': 0,
            'rs10007052': 1.5, 'rs8192288': 1.2, 'rs20541': 1.8,
            'rs12922394': 1.1, 'rs2910164': 1.3, 'rs161976': 1.4,
            'rs473892': 1.6, 'rs159497': 1.2, 'rs9296092': 1.5
        },
        # Test case 8: Young overweight smoker, mixed genetic marker values
        {
            'age': 30, 'bmi': 27.0, 'smoke': 1,
            'rs10007052': 1.2, 'rs8192288': 0.9, 'rs20541': 1.4,
            'rs12922394': 1.1, 'rs2910164': 1.0, 'rs161976': 1.5,
            'rs473892': 1.3, 'rs159497': 0.8, 'rs9296092': 1.2
        },
        # Missing values for some features
        {
            'age': None, 'bmi': 25.0, 'smoke': 1,
            'rs10007052': 1.1, 'rs8192288': None, 'rs20541': 1.3,
            'rs12922394': 1.5, 'rs2910164': 1.2, 'rs161976': None,
            'rs473892': 1.0, 'rs159497': None, 'rs9296092': 1.1
        },
        # Out-of-range values
        {
            'age': -5, 'bmi': 400.0, 'smoke': 0,
            'rs10007052': 3.0, 'rs8192288': 3.0, 'rs20541': 3.0,
            'rs12922394': 3.0, 'rs2910164': 3.0, 'rs161976': 3.0,
            'rs473892': 3.0, 'rs159497': 3.0, 'rs9296092': 3.0
        },
        # Incorrect data types
        {
            'age': "thirty", 'bmi': "twenty-five", 'smoke': "yes",
            'rs10007052': "one", 'rs8192288': "two", 'rs20541': "three",
            'rs12922394': "four", 'rs2910164': "five", 'rs161976': "six",
            'rs473892': "seven", 'rs159497': "eight", 'rs9296092': "nine"
        },
        # Negative values
        {
            'age': 50, 'bmi': -22.0, 'smoke': 1,
            'rs10007052': -1.0, 'rs8192288': -1.0, 'rs20541': -1.0,
            'rs12922394': -1.0, 'rs2910164': -1.0, 'rs161976': -1.0,
            'rs473892': -1.0, 'rs159497': -1.0, 'rs9296092': -1.0
        },
        # Unreasonable combinations
        {
            'age': 5, 'bmi': 18.0, 'smoke': 1,
            'rs10007052': 1.0, 'rs8192288': 1.0, 'rs20541': 1.0,
            'rs12922394': 1.0, 'rs2910164': 1.0, 'rs161976': 1.0,
            'rs473892': 1.0, 'rs159497': 1.0, 'rs9296092': 1.0
        }
    ])


def ensure_feature_alignment(data, feature_columns):
    for col in feature_columns:
        if col not in data.columns:
            data[col] = None  # Add missing columns with default `None`
        else:
            data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric

    return data


def validate_and_impute_data(data, imputer):
    """Validate and impute missing data in test cases."""
    missing_data = data.isnull().sum()
    if missing_data.any():
        print("Warning: Missing values detected in the following columns:")
        print(missing_data[missing_data > 0])

    # Impute missing values in genetic marker columns
    try:
        data[GENETIC_MARKER_COLUMNS] = imputer.transform(data[GENETIC_MARKER_COLUMNS])
    except NotFittedError:
        print("Imputer is not fitted. Please recheck the preprocessing step.")
        raise

    return data


def make_predictions(model, label_encoder, test_data):
    """Make predictions using the loaded model and encoder."""
    predictions = model.predict(test_data)
    predicted_risk_levels = label_encoder.inverse_transform(predictions)
    return predicted_risk_levels


def explain_predictions(model, test_data, predictions):
    """Generate SHAP explanations for predictions."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_data)

        for i, test_case in test_data.iterrows():
            predicted_class_index = predictions[i]
            base_value = explainer.expected_value[predicted_class_index]
            shap_values_for_case = shap_values[i, :, predicted_class_index]

            print(f"Test case {i + 1}:")
            print(f"Predicted risk level: {predicted_class_index}")
            print(f"Base value: {base_value}")
            print(f"SHAP values: {shap_values_for_case}")

            shap.force_plot(
                base_value,
                shap_values_for_case,
                test_case,
                matplotlib=True
            )
    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")


if __name__ == "__main__":
    try:
        # Load saved model components
        with open("model_components.pkl", "rb") as f:
            model, label_encoder, imputer = pickle.load(f)

        # Generate test cases
        test_data = create_test_data()

        # Align test data with features and validate
        feature_columns = ['age', 'bmi', 'smoke'] + GENETIC_MARKER_COLUMNS
        test_data = ensure_feature_alignment(test_data, feature_columns)
        test_data = validate_and_impute_data(test_data, imputer)

        # Make predictions
        predicted_risk_levels = make_predictions(model, label_encoder, test_data)

        # Output predictions
        for i, risk_level in enumerate(predicted_risk_levels):
            print(f"Test case {i + 1}: Predicted COPD risk level: {risk_level}")

        # Generate SHAP explanations
        explain_predictions(model, test_data, predicted_risk_levels)

    except FileNotFoundError:
        print("Error: Model components file not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
