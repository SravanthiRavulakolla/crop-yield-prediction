from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model and scalers
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')

# Load data for label encoders and feature names
df_original = pd.read_csv('crop.csv')
df_engineered = pd.read_csv('crop_features.csv')

# Get target and features
target = 'Crop Yield'
all_features = [col for col in df_engineered.columns if col != target]

# Fit label encoders
df_test = df_engineered.copy()
categorical_cols = df_test.select_dtypes(include='object').columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df_test[col])
    label_encoders[col] = le

# Get unique values for dropdowns
soil_types = sorted(df_original['Soil Type'].unique().tolist())
crop_types = sorted(df_original['Crop Type'].unique().tolist())
fertilizers = sorted(df_original['Fertilizer Name'].unique().tolist())

@app.route('/')
def index():
    return render_template('index.html',
                         soil_types=soil_types,
                         crop_types=crop_types,
                         fertilizers=fertilizers)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract inputs
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        moisture = float(data['moisture'])
        soil_type = data['soil_type']
        crop_type = data['crop_type']
        nitrogen = float(data['nitrogen'])
        phosphorous = float(data['phosphorous'])
        potassium = float(data['potassium'])
        fertilizer = data['fertilizer']
        
        # Create raw data dictionary
        raw_data = {
            'Temparature': temp,
            'Humidity': humidity,
            'Moisture': moisture,
            'Nitrogen': nitrogen,
            'Potassium': potassium,
            'Phosphorous': phosphorous,
            'Crop Yield': 0  # Dummy value
        }
        
        # Scale numerical features
        scaler_features = scaler.get_feature_names_out().tolist()
        raw_df_for_scaler = pd.DataFrame([[
            raw_data['Temparature'],
            raw_data['Humidity'],
            raw_data['Moisture'],
            raw_data['Nitrogen'],
            raw_data['Potassium'],
            raw_data['Phosphorous'],
            0
        ]], columns=scaler_features)
        
        scaled_array = scaler.transform(raw_df_for_scaler)
        scaled_df = pd.DataFrame(scaled_array, columns=scaler_features)
        
        # Encode categorical features
        soil_encoded = label_encoders['Soil Type'].transform([soil_type])[0]
        crop_encoded = label_encoders['Crop Type'].transform([crop_type])[0]
        fert_encoded = label_encoders['Fertilizer Name'].transform([fertilizer])[0]
        
        # Build scaled data dictionary
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
        
        # Compute interaction features
        scaled_data['N_P_interaction'] = scaled_data['Nitrogen'] * scaled_data['Phosphorous']
        scaled_data['Temp_Humidity_interaction'] = scaled_data['Temparature'] * scaled_data['Humidity']
        scaled_data['K_N_interaction'] = scaled_data['Potassium'] * scaled_data['Nitrogen']
        
        # Compute ratio features
        scaled_data['N_to_P_ratio'] = (scaled_data['Nitrogen'] / scaled_data['Phosphorous']) if scaled_data['Phosphorous'] != 0 else 0
        scaled_data['K_to_P_ratio'] = (scaled_data['Potassium'] / scaled_data['Phosphorous']) if scaled_data['Phosphorous'] != 0 else 0
        
        # Compute average yield for crop type
        avg_yield = df_engineered[df_engineered['Crop Type'] == crop_type][target].mean()
        scaled_data['Avg_Yield_CropType'] = avg_yield if not pd.isna(avg_yield) else 0
        
        # Prepare input for model
        input_df = pd.DataFrame([scaled_data])[all_features]
        input_df = input_df.fillna(0)
        
        # Make prediction
        predicted_yield_scaled = model.predict(input_df)[0]
        predicted_yield = target_scaler.inverse_transform([[predicted_yield_scaled]])[0][0]
        
        return jsonify({
            'success': True,
            'prediction': round(predicted_yield, 2),
            'message': f'Predicted Crop Yield: {predicted_yield:.2f} kg/hectare'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

