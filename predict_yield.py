import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load best model
try:
    model = load_model('best_model.h5')
    model_type = 'ANN'
except:
    model = joblib.load('best_model.pkl')
    model_type = 'Linear Regression'

# Load original unscaled data
df_original = pd.read_csv('crop.csv')
df_original.columns = df_original.columns.str.strip()

# Load engineered features data to get structure
df_engineered = pd.read_csv('crop_features.csv')
df_engineered.columns = df_engineered.columns.str.strip()

# Get target column name
target = [col for col in df_engineered.columns if 'yield' in col.lower()][0]

# Get all feature names that the model expects
all_features = [col for col in df_engineered.columns if col != target]

# Get numerical features from original data (before engineering)
numerical_features_original = df_original.select_dtypes(include=['int64','float64']).columns.tolist()
if target in numerical_features_original:
    numerical_features_original.remove(target)

# Load the saved scalers from preprocess.py
scaler = joblib.load('scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')

# Encode categorical variables from original data
categorical_cols = df_original.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df_original[col])
    label_encoders[col] = le

print("\n--- Crop Yield Prediction ---")

# ---- Get basic user inputs ----
temp = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
moisture = float(input("Enter Moisture: "))
soil_type = input(f"Enter Soil Type (options: {list(label_encoders['Soil Type'].classes_)}): ").strip()
crop_type = input(f"Enter Crop Type (options: {list(label_encoders['Crop Type'].classes_)}): ").strip()
N = float(input("Enter Nitrogen value: "))
P = float(input("Enter Phosphorous value: "))
K = float(input("Enter Potassium value: "))
fertilizer = input(f"Enter Fertilizer Name (options: {list(label_encoders['Fertilizer Name'].classes_)}): ").strip()

# ---- Encode categorical ----
soil_encoded = label_encoders['Soil Type'].transform([soil_type if soil_type in label_encoders['Soil Type'].classes_ else label_encoders['Soil Type'].classes_[0]])[0]
crop_encoded = label_encoders['Crop Type'].transform([crop_type if crop_type in label_encoders['Crop Type'].classes_ else label_encoders['Crop Type'].classes_[0]])[0]
fert_encoded = label_encoders['Fertilizer Name'].transform([fertilizer if fertilizer in label_encoders['Fertilizer Name'].classes_ else label_encoders['Fertilizer Name'].classes_[0]])[0]

# ---- Create input data with raw values ----
raw_data = {
    'Temparature': temp,
    'Humidity': humidity,
    'Moisture': moisture,
    'Nitrogen': N,
    'Phosphorous': P,
    'Potassium': K
}

# ---- Create DataFrame and scale all numerical features at once ----
# The scaler was fit on all numerical features including Crop Yield
# We need to provide data in the same order as the scaler expects
scaler_features = scaler.get_feature_names_out().tolist()

raw_df_for_scaler = pd.DataFrame([[
    raw_data['Temparature'],
    raw_data['Humidity'],
    raw_data['Moisture'],
    raw_data['Nitrogen'],
    raw_data['Potassium'],
    raw_data['Phosphorous'],
    0  # Dummy Crop Yield
]], columns=scaler_features)

# Scale using the saved scaler
scaled_array = scaler.transform(raw_df_for_scaler)
scaled_df = pd.DataFrame(scaled_array, columns=scaler_features)

# Keep only the features we need (without Crop Yield)
scaled_df = scaled_df[numerical_features_original]

# ---- Build complete feature dictionary ----
scaled_data = {
    'Temparature': scaled_df['Temparature'].iloc[0],
    'Humidity': scaled_df['Humidity'].iloc[0],
    'Moisture': scaled_df['Moisture'].iloc[0],
    'Soil Type': soil_encoded,
    'Crop Type': crop_encoded,
    'Nitrogen': scaled_df['Nitrogen'].iloc[0],
    'Phosphorous': scaled_df['Phosphorous'].iloc[0],
    'Potassium': scaled_df['Potassium'].iloc[0],
    'Fertilizer Name': fert_encoded
}

# ---- Compute interaction features (using scaled values) ----
scaled_data['N_P_interaction'] = scaled_data['Nitrogen'] * scaled_data['Phosphorous']
scaled_data['Temp_Humidity_interaction'] = scaled_data['Temparature'] * scaled_data['Humidity']
scaled_data['K_N_interaction'] = scaled_data['Potassium'] * scaled_data['Nitrogen']

# ---- Compute ratio features (using scaled values) ----
scaled_data['N_to_P_ratio'] = (scaled_data['Nitrogen'] / scaled_data['Phosphorous']) if scaled_data['Phosphorous'] != 0 else 0
scaled_data['K_to_P_ratio'] = (scaled_data['Potassium'] / scaled_data['Phosphorous']) if scaled_data['Phosphorous'] != 0 else 0

# ---- Compute average yield for crop type ----
# Use the string crop type to filter, not the encoded value
avg_yield = df_engineered[df_engineered['Crop Type'] == crop_type][target].mean()
scaled_data['Avg_Yield_CropType'] = avg_yield if not pd.isna(avg_yield) else 0

# ---- Prepare final DataFrame with correct feature order ----
input_df = pd.DataFrame([scaled_data])[all_features]

# Fill any NaN values with 0
input_df = input_df.fillna(0)

# ---- Predict ----
if model_type == 'ANN':
    predicted_yield_scaled = model.predict(input_df, verbose=0)[0][0]
else:
    predicted_yield_scaled = model.predict(input_df)[0]

# ---- Inverse scale the prediction to get actual yield ----
predicted_yield = target_scaler.inverse_transform([[predicted_yield_scaled]])[0][0]

print(f"\nPredicted Crop Yield: {predicted_yield:.2f} kg/hectare")
print(f"Prediction done using {model_type} model.")
