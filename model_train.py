import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load datasets
file_path = "C:/Users/shiva/Downloads/Finalalldata.csv"
data = pd.read_csv(file_path)
asthma_data = pd.read_csv("C:/Users/shiva/Downloads/asthma_dataset.csv")

# Prepare asthma_data
asthma_data = asthma_data.rename(columns={
    'Patient_ID': 'uid',
    'Age': 'age',
    'Gender': 'sex',
    'Smoking_Status': 'smoke'
})
asthma_data['sex'] = asthma_data['sex'].map({'Male': 1, 'Female': 2})
asthma_data['smoke'] = asthma_data['smoke'].map({'Yes': 1, 'No': 0})

# Add placeholder columns for missing genetic markers and label
genetic_marker_columns = ['rs10007052', 'rs8192288', 'rs20541', 'rs12922394',
                          'rs2910164', 'rs161976', 'rs473892', 'rs159497', 'rs9296092']
for col in genetic_marker_columns:
    asthma_data[col] = pd.NA

asthma_data['risk_level'] = 'High Risk'
asthma_data.drop(columns=['Medication', 'Peak_Flow'], inplace=True, errors='ignore')

# Combine datasets
combined_data = pd.concat([data, asthma_data], ignore_index=True)

# Ensure all genetic_marker_columns exist in combined_data before fitting imputer
for col in genetic_marker_columns:
    if col not in combined_data.columns:
        combined_data[col] = np.nan

# Impute missing values in genetic marker columns with median values
imputer = SimpleImputer(strategy="median")
combined_data[genetic_marker_columns] = imputer.fit_transform(combined_data[genetic_marker_columns])


# Function to assign risk level
def assign_risk_level(row):
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


combined_data['risk_level'] = combined_data.apply(assign_risk_level, axis=1)

# Prepare features and target
features = combined_data[['age', 'bmi', 'smoke'] + genetic_marker_columns]
target = combined_data['risk_level']
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)

# SHAP Integration
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# User input for prediction
user_data = pd.DataFrame([{
    'age': 50,
    'bmi': 28.5,
    'smoke': 1,
    'rs10007052': None,
    'rs8192288': 1.5,
    'rs20541': 2.1,
    'rs12922394': None,
    'rs2910164': 1.8,
    'rs161976': 2.5,
    'rs473892': 1.2,
    'rs159497': 0.5,
    'rs9296092': None
}])

# Align user_data with feature columns
for col in features.columns:
    if col not in user_data.columns:
        user_data[col] = np.nan
user_data = user_data[features.columns]

# Impute missing values in user data
user_data[genetic_marker_columns] = imputer.transform(user_data[genetic_marker_columns])
class_labels = label_encoder.classes_

# Predict and explain
user_prediction = model.predict(user_data)
predicted_risk_level = label_encoder.inverse_transform(user_prediction)
selected_class_index = list(class_labels).index(predicted_risk_level)
print(f"Predicted COPD risk level for user: {predicted_risk_level[0]}")

user_data_reshaped = user_data.iloc[0:1]
print("user_data shape:", user_data.shape)
shap_values_user = explainer.shap_values(user_data_reshaped)
print("SHAP values shape:", np.array(shap_values_user).shape)

shap.force_plot(
    explainer.expected_value[selected_class_index],  # Expected value for the selected class
    shap_values_user[0][:, selected_class_index],    # SHAP values for the selected class
    user_data.iloc[0],                               # Feature values
    matplotlib=True
)
