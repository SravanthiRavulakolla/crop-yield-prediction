
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

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

df.to_csv('crop_scaled.csv', index=False)
print("Missing values handled, outliers addressed, numerical features standardized, and dataset saved as 'crop_scaled.csv'")
