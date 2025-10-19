import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib

# ----------------------------
# Step 1: Load dataset
# ----------------------------
df = pd.read_csv('crop_features.csv')
df.columns = df.columns.str.strip()

# ----------------------------
# Step 2: Encode all categorical variables
# ----------------------------
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ----------------------------
# Step 3: Define features & target
# ----------------------------
target = [col for col in df.columns if 'yield' in col.lower()][0]
X = df.drop(columns=[target])
y = df[target]

# ----------------------------
# Step 4: Split data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Step 5: Train Linear Regression
# ----------------------------
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)

reg_r2 = r2_score(y_test, y_pred_reg)
reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
reg_mae = mean_absolute_error(y_test, y_pred_reg)

# ----------------------------
# Step 6: Train ANN Model
# ----------------------------
ann = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1)
])

ann.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
ann.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

y_pred_ann = ann.predict(X_test).flatten()

ann_r2 = r2_score(y_test, y_pred_ann)
ann_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ann))
ann_mae = mean_absolute_error(y_test, y_pred_ann)

# ----------------------------
# Step 7: Compare Results
# ----------------------------
results = pd.DataFrame({
    'Model': ['Linear Regression', 'ANN'],
    'R2 Score': [reg_r2, ann_r2],
    'RMSE': [reg_rmse, ann_rmse],
    'MAE': [reg_mae, ann_mae]
})

print("\nModel Performance Comparison:\n")
print(results)

results.to_csv('model_results.csv', index=False)

# ----------------------------
# Step 8: Save Best Model
# ----------------------------
best_model = 'ANN' if ann_r2 > reg_r2 else 'Linear Regression'

if best_model == 'ANN':
    ann.save('best_model.h5')
else:
    joblib.dump(reg, 'best_model.pkl')

print(f"\n✅ Best model: {best_model} saved successfully!")
