import pandas as pd

# ----------------------------
# Step 1: Load preprocessed dataset
# ----------------------------
df = pd.read_csv('crop_scaled.csv')
df.columns = df.columns.str.strip()

# ----------------------------
# Step 2: Define important features
# ----------------------------
important_numerical = ['Potassium', 'Nitrogen', 'Humidity', 'Temparature', 'Phosphorous']
important_categorical = ['Crop Type']
target = [col for col in df.columns if 'yield' in col.lower()][0]

# ----------------------------
# Step 3: Create Interaction Features (Numerical × Numerical)
# ----------------------------
df['N_P_interaction'] = df['Nitrogen'] * df['Phosphorous']
df['Temp_Humidity_interaction'] = df['Temparature'] * df['Humidity']
df['K_N_interaction'] = df['Potassium'] * df['Nitrogen']

# ----------------------------
# Step 4: Create Ratio Features
# ----------------------------
df['N_to_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
df['K_to_P_ratio'] = df['Potassium'] / (df['Phosphorous'] + 1e-6)

# ----------------------------
# Step 5: Categorical Aggregates
# ----------------------------
df['Avg_Yield_CropType'] = df.groupby('Crop Type')[target].transform('mean')

# ----------------------------
# Step 6: Save final dataset
# ----------------------------
df.to_csv('crop_features.csv', index=False)
print("Feature engineering completed. Final dataset saved as 'crop_features.csv'")