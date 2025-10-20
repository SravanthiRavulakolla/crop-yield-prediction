
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('crop.csv')
df.columns = df.columns.str.strip()

numerical_features = df.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = df.select_dtypes(include='object').columns.tolist()

for col in numerical_features:
    df[col].fillna(df[col].median(), inplace=True)

for col in categorical_features:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numerical_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

# Save the clipped data before scaling (for target scaler)
df_clipped = df.copy()

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

df.to_csv('crop_scaled.csv', index=False)

# Save the scaler for later use in predictions
import joblib
joblib.dump(scaler, 'scaler.pkl')

# Also save the target scaler parameters for inverse transformation
# We need to fit it on the CLIPPED data (after outlier removal) to match the training process
target_scaler = StandardScaler()
target_scaler.fit(df_clipped[['Crop Yield']])
joblib.dump(target_scaler, 'target_scaler.pkl')

print("Missing values handled, outliers addressed, numerical features standardized, and dataset saved as 'crop_scaled.csv'")
print("Scaler saved as 'scaler.pkl'")
print("Target scaler saved as 'target_scaler.pkl'")
