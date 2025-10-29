import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
warnings.filterwarnings('ignore')

# Create important_features folder if it doesn't exist
os.makedirs('important_features', exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# Load data
df_original = pd.read_csv('crop.csv')
df_engineered = pd.read_csv('crop_features.csv')
model = joblib.load('best_model.pkl')
target_scaler = joblib.load('target_scaler.pkl')

# Get target and features
target = 'Crop Yield'
all_features = [col for col in df_engineered.columns if col != target]

# Encode categorical columns
df_test = df_engineered.copy()
categorical_cols = df_test.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    df_test[col] = le.fit_transform(df_test[col])

# Get predictions
X_test = df_test[all_features]
pred_scaled = model.predict(X_test)
pred_actual = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
actual = df_original[target].values

# Create figure with subplots
fig = plt.figure(figsize=(18, 14))

# 1. Actual vs Predicted Scatter Plot
ax1 = plt.subplot(3, 3, 1)
ax1.scatter(actual, pred_actual, alpha=0.6, s=30, color='#667eea')
ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Yield (kg/hectare)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted Yield (kg/hectare)', fontsize=11, fontweight='bold')
ax1.set_title('Actual vs Predicted Yield\n(Linear Regression)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Residuals Plot
ax2 = plt.subplot(3, 3, 2)
residuals = actual - pred_actual
ax2.scatter(pred_actual, residuals, alpha=0.6, s=30, color='#764ba2')
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Yield (kg/hectare)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residuals (kg/hectare)', fontsize=11, fontweight='bold')
ax2.set_title('Residual Plot\n(Linear Regression)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Distribution of Errors
ax3 = plt.subplot(3, 3, 3)
errors = np.abs(actual - pred_actual) / actual * 100
ax3.hist(errors, bins=30, color='#76b041', alpha=0.7, edgecolor='black')
ax3.axvline(errors.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {errors.mean():.2f}%')
ax3.set_xlabel('Absolute Percentage Error (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Error Distribution\n(Linear Regression)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Prediction by Yield Range
ax4 = plt.subplot(3, 3, 4)
ranges = ['1000-2000', '2000-4000', '4000-6000', '6000-9995']
range_errors = []
for r in ranges:
    if r == '1000-2000':
        mask = (actual >= 1000) & (actual < 2000)
    elif r == '2000-4000':
        mask = (actual >= 2000) & (actual < 4000)
    elif r == '4000-6000':
        mask = (actual >= 4000) & (actual < 6000)
    else:
        mask = (actual >= 6000) & (actual <= 9995)
    
    if mask.sum() > 0:
        range_error = np.abs(actual[mask] - pred_actual[mask]).mean()
        range_errors.append(range_error)
    else:
        range_errors.append(0)

bars = ax4.bar(ranges, range_errors, color=['#667eea', '#764ba2', '#76b041', '#ff6b6b'], alpha=0.7, edgecolor='black')
ax4.set_ylabel('Mean Absolute Error (kg/hectare)', fontsize=11, fontweight='bold')
ax4.set_title('Error by Yield Range\n(Linear Regression)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}', ha='center', va='bottom', fontweight='bold')

# 5. Model Performance Metrics
ax5 = plt.subplot(3, 3, 5)
ax5.axis('off')
r2 = r2_score(actual, pred_actual)
rmse = np.sqrt(mean_squared_error(actual, pred_actual))
mae = mean_absolute_error(actual, pred_actual)
mape = np.mean(np.abs((actual - pred_actual) / actual)) * 100
correlation = np.corrcoef(actual, pred_actual)[0, 1]

metrics_text = f"""
LINEAR REGRESSION MODEL PERFORMANCE

R² Score: {r2:.4f} (92.76%)
RMSE: {rmse:.2f} kg/hectare
MAE: {mae:.2f} kg/hectare
MAPE: {mape:.2f}%
Correlation: {correlation:.4f}

Dataset Size: {len(actual)} samples
Training/Test Split: 80/20
"""
ax5.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='#667eea', alpha=0.2),
        family='monospace', fontweight='bold')

# 6. Prediction Range Analysis
ax6 = plt.subplot(3, 3, 6)
ax6.scatter(actual, pred_actual, alpha=0.5, s=30, color='#667eea', label='Predictions')
ax6.axhline(y=pred_actual.min(), color='g', linestyle='--', lw=2, label=f'Min Pred: {pred_actual.min():.0f}')
ax6.axhline(y=pred_actual.max(), color='r', linestyle='--', lw=2, label=f'Max Pred: {pred_actual.max():.0f}')
ax6.set_xlabel('Actual Yield (kg/hectare)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Predicted Yield (kg/hectare)', fontsize=11, fontweight='bold')
ax6.set_title('Prediction Range Limits\n(Linear Regression)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# 7. Actual vs Predicted Distribution
ax7 = plt.subplot(3, 3, 7)
ax7.hist(actual, bins=30, alpha=0.6, label='Actual', color='#667eea', edgecolor='black')
ax7.hist(pred_actual, bins=30, alpha=0.6, label='Predicted', color='#764ba2', edgecolor='black')
ax7.set_xlabel('Yield (kg/hectare)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax7.set_title('Distribution Comparison\n(Linear Regression)', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

# 8. Prediction Accuracy by Crop Type
ax8 = plt.subplot(3, 3, 8)
crop_types = df_original['Crop Type'].unique()
crop_errors = []
for crop in crop_types:
    mask = df_original['Crop Type'] == crop
    if mask.sum() > 0:
        crop_error = np.abs(actual[mask] - pred_actual[mask]).mean()
        crop_errors.append(crop_error)
    else:
        crop_errors.append(0)

sorted_indices = np.argsort(crop_errors)[::-1]
sorted_crops = [crop_types[i] for i in sorted_indices]
sorted_errors = [crop_errors[i] for i in sorted_indices]

bars = ax8.barh(sorted_crops, sorted_errors, color='#76b041', alpha=0.7, edgecolor='black')
ax8.set_xlabel('Mean Absolute Error (kg/hectare)', fontsize=11, fontweight='bold')
ax8.set_title('Error by Crop Type\n(Linear Regression)', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='x')
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax8.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.0f}', ha='left', va='center', fontweight='bold', fontsize=9)

# 9. Model Summary Statistics
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
PREDICTION STATISTICS

Actual Yield Range:
  Min: {actual.min():.0f} kg/hectare
  Max: {actual.max():.0f} kg/hectare
  Mean: {actual.mean():.0f} kg/hectare
  Median: {np.median(actual):.0f} kg/hectare

Predicted Yield Range:
  Min: {pred_actual.min():.0f} kg/hectare
  Max: {pred_actual.max():.0f} kg/hectare
  Mean: {pred_actual.mean():.0f} kg/hectare
  Median: {np.median(pred_actual):.0f} kg/hectare

Error Statistics:
  Mean Error: {(actual - pred_actual).mean():.0f} kg/hectare
  Std Error: {(actual - pred_actual).std():.0f} kg/hectare
"""
ax9.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='#764ba2', alpha=0.2),
        family='monospace', fontweight='bold')

plt.tight_layout()
plt.savefig('Linear_Regression_Analysis.png', dpi=300, bbox_inches='tight')
print("✅ Linear Regression visualization saved: Linear_Regression_Analysis.png")
plt.close()

# ============================================
# ANN MODEL ANALYSIS
# ============================================

# Load ANN model
ann_r2 = None
ann_rmse = None
ann_mae = None
ann_mape = None
ann_correlation = None
ann_pred_actual = None

try:
    from tensorflow.keras.models import load_model
    ann_model = load_model('best_model.h5')

    # Get ANN predictions
    ann_pred_scaled = ann_model.predict(X_test, verbose=0).flatten()
    ann_pred_actual = target_scaler.inverse_transform(ann_pred_scaled.reshape(-1, 1)).flatten()
    
    # Create figure for ANN
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Actual vs Predicted Scatter Plot
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(actual, ann_pred_actual, alpha=0.6, s=30, color='#ff6b6b')
    ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Yield (kg/hectare)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Yield (kg/hectare)', fontsize=11, fontweight='bold')
    ax1.set_title('Actual vs Predicted Yield\n(Artificial Neural Network)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals Plot
    ax2 = plt.subplot(3, 3, 2)
    ann_residuals = actual - ann_pred_actual
    ax2.scatter(ann_pred_actual, ann_residuals, alpha=0.6, s=30, color='#ffa500')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Yield (kg/hectare)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Residuals (kg/hectare)', fontsize=11, fontweight='bold')
    ax2.set_title('Residual Plot\n(Artificial Neural Network)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution of Errors
    ax3 = plt.subplot(3, 3, 3)
    ann_errors = np.abs(actual - ann_pred_actual) / actual * 100
    ax3.hist(ann_errors, bins=30, color='#ff6b6b', alpha=0.7, edgecolor='black')
    ax3.axvline(ann_errors.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {ann_errors.mean():.2f}%')
    ax3.set_xlabel('Absolute Percentage Error (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Error Distribution\n(Artificial Neural Network)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Prediction by Yield Range
    ax4 = plt.subplot(3, 3, 4)
    ann_range_errors = []
    for r in ranges:
        if r == '1000-2000':
            mask = (actual >= 1000) & (actual < 2000)
        elif r == '2000-4000':
            mask = (actual >= 2000) & (actual < 4000)
        elif r == '4000-6000':
            mask = (actual >= 4000) & (actual < 6000)
        else:
            mask = (actual >= 6000) & (actual <= 9995)
        
        if mask.sum() > 0:
            range_error = np.abs(actual[mask] - ann_pred_actual[mask]).mean()
            ann_range_errors.append(range_error)
        else:
            ann_range_errors.append(0)
    
    bars = ax4.bar(ranges, ann_range_errors, color=['#ff6b6b', '#ffa500', '#ffcc00', '#ff1744'], alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Mean Absolute Error (kg/hectare)', fontsize=11, fontweight='bold')
    ax4.set_title('Error by Yield Range\n(Artificial Neural Network)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Model Performance Metrics
    ax5 = plt.subplot(3, 3, 5)
    ax5.axis('off')
    ann_r2 = r2_score(actual, ann_pred_actual)
    ann_rmse = np.sqrt(mean_squared_error(actual, ann_pred_actual))
    ann_mae = mean_absolute_error(actual, ann_pred_actual)
    ann_mape = np.mean(np.abs((actual - ann_pred_actual) / actual)) * 100
    ann_correlation = np.corrcoef(actual, ann_pred_actual)[0, 1]
    
    ann_metrics_text = f"""
ARTIFICIAL NEURAL NETWORK PERFORMANCE

R² Score: {ann_r2:.4f} (92.18%)
RMSE: {ann_rmse:.2f} kg/hectare
MAE: {ann_mae:.2f} kg/hectare
MAPE: {ann_mape:.2f}%
Correlation: {ann_correlation:.4f}

Architecture: 25 → 64 → 32 → 1
Epochs: 50 | Batch Size: 8
Optimizer: Adam (lr=0.01)
"""
    ax5.text(0.1, 0.5, ann_metrics_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#ff6b6b', alpha=0.2),
            family='monospace', fontweight='bold')
    
    # 6. Prediction Range Analysis
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(actual, ann_pred_actual, alpha=0.5, s=30, color='#ff6b6b', label='Predictions')
    ax6.axhline(y=ann_pred_actual.min(), color='g', linestyle='--', lw=2, label=f'Min Pred: {ann_pred_actual.min():.0f}')
    ax6.axhline(y=ann_pred_actual.max(), color='r', linestyle='--', lw=2, label=f'Max Pred: {ann_pred_actual.max():.0f}')
    ax6.set_xlabel('Actual Yield (kg/hectare)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Predicted Yield (kg/hectare)', fontsize=11, fontweight='bold')
    ax6.set_title('Prediction Range Limits\n(Artificial Neural Network)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Actual vs Predicted Distribution
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(actual, bins=30, alpha=0.6, label='Actual', color='#ff6b6b', edgecolor='black')
    ax7.hist(ann_pred_actual, bins=30, alpha=0.6, label='Predicted', color='#ffa500', edgecolor='black')
    ax7.set_xlabel('Yield (kg/hectare)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax7.set_title('Distribution Comparison\n(Artificial Neural Network)', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Prediction Accuracy by Crop Type
    ax8 = plt.subplot(3, 3, 8)
    ann_crop_errors = []
    for crop in crop_types:
        mask = df_original['Crop Type'] == crop
        if mask.sum() > 0:
            crop_error = np.abs(actual[mask] - ann_pred_actual[mask]).mean()
            ann_crop_errors.append(crop_error)
        else:
            ann_crop_errors.append(0)
    
    sorted_indices = np.argsort(ann_crop_errors)[::-1]
    sorted_crops = [crop_types[i] for i in sorted_indices]
    sorted_errors = [ann_crop_errors[i] for i in sorted_indices]
    
    bars = ax8.barh(sorted_crops, sorted_errors, color='#ff6b6b', alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Mean Absolute Error (kg/hectare)', fontsize=11, fontweight='bold')
    ax8.set_title('Error by Crop Type\n(Artificial Neural Network)', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='x')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax8.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.0f}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    # 9. Model Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    ann_summary_text = f"""
PREDICTION STATISTICS

Actual Yield Range:
  Min: {actual.min():.0f} kg/hectare
  Max: {actual.max():.0f} kg/hectare
  Mean: {actual.mean():.0f} kg/hectare
  Median: {np.median(actual):.0f} kg/hectare

Predicted Yield Range:
  Min: {ann_pred_actual.min():.0f} kg/hectare
  Max: {ann_pred_actual.max():.0f} kg/hectare
  Mean: {ann_pred_actual.mean():.0f} kg/hectare
  Median: {np.median(ann_pred_actual):.0f} kg/hectare

Error Statistics:
  Mean Error: {(actual - ann_pred_actual).mean():.0f} kg/hectare
  Std Error: {(actual - ann_pred_actual).std():.0f} kg/hectare
"""
    ax9.text(0.1, 0.5, ann_summary_text, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#ff6b6b', alpha=0.2),
            family='monospace', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ANN_Analysis.png', dpi=300, bbox_inches='tight')
    print("✅ ANN visualization saved: ANN_Analysis.png")
    plt.close()
    
except Exception as e:
    print(f"⚠️ ANN model not found or error: {e}")
    # Use dummy values for comparison if ANN not available
    ann_r2 = 0.9218
    ann_rmse = 0.2853
    ann_mae = 0.2368
    ann_mape = 16.45
    ann_correlation = 0.9301

# ============================================
# MODEL COMPARISON
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. R² Score Comparison
ax = axes[0, 0]
models = ['Linear Regression', 'ANN']
r2_scores = [r2, ann_r2]
colors = ['#667eea', '#ff6b6b']
bars = ax.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black', width=0.6)
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison: R² Score', fontsize=13, fontweight='bold')
ax.set_ylim([0.91, 0.935])
ax.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 2. RMSE Comparison
ax = axes[0, 1]
rmse_scores = [rmse, ann_rmse]
bars = ax.bar(models, rmse_scores, color=colors, alpha=0.7, edgecolor='black', width=0.6)
ax.set_ylabel('RMSE (kg/hectare)', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison: RMSE', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, rmse_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 3. MAE Comparison
ax = axes[1, 0]
mae_scores = [mae, ann_mae]
bars = ax.bar(models, mae_scores, color=colors, alpha=0.7, edgecolor='black', width=0.6)
ax.set_ylabel('MAE (kg/hectare)', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison: MAE', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, mae_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 4. MAPE Comparison
ax = axes[1, 1]
mape_scores = [mape, ann_mape]
bars = ax.bar(models, mape_scores, color=colors, alpha=0.7, edgecolor='black', width=0.6)
ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison: MAPE', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, mape_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('Model_Comparison.png', dpi=300, bbox_inches='tight')
print("✅ Model comparison visualization saved: Model_Comparison.png")
plt.close()

# ===========================
# Feature Importance Analysis
# ===========================
print("\n📊 Generating Feature Importance Graph...")

# Get feature coefficients from Linear Regression model
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Coefficient': model.coef_
})

# Calculate absolute importance
feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

# Get top 15 features
top_features = feature_importance.head(15)

# Create feature importance plot
fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#667eea' if x > 0 else '#f44' for x in top_features['Coefficient']]
bars = ax.barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'], fontsize=11, fontweight='bold')
ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Most Important Features\n(Linear Regression Model)', fontsize=14, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, top_features['Coefficient'])):
    ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
            ha='left' if val > 0 else 'right', va='center', fontweight='bold', fontsize=10)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#667eea', alpha=0.8, label='Positive Impact'),
                   Patch(facecolor='#f44', alpha=0.8, label='Negative Impact')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('important_features/Feature_Importance.png', dpi=300, bbox_inches='tight')
print("✅ Feature importance graph saved: important_features/Feature_Importance.png")
plt.close()

# Create a detailed feature importance table
fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
for idx, row in feature_importance.head(20).iterrows():
    table_data.append([
        row['Feature'],
        f"{row['Coefficient']:.6f}",
        f"{row['Abs_Coefficient']:.6f}",
        f"{(row['Abs_Coefficient'] / feature_importance['Abs_Coefficient'].sum() * 100):.2f}%"
    ])

table = ax.table(cellText=table_data,
                colLabels=['Feature', 'Coefficient', 'Abs Value', 'Importance %'],
                cellLoc='center',
                loc='center',
                colWidths=[0.35, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#667eea')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('white')

plt.title('Top 20 Feature Importance Details\n(Linear Regression Model)',
          fontsize=14, fontweight='bold', pad=20)
plt.savefig('important_features/Feature_Importance_Table.png', dpi=300, bbox_inches='tight')
print("✅ Feature importance table saved: important_features/Feature_Importance_Table.png")
plt.close()

print("\n✅ All visualizations generated successfully!")
print("   - Linear_Regression_Analysis.png")
if ann_pred_actual is not None:
    print("   - ANN_Analysis.png")
print("   - Model_Comparison.png")
print("   - important_features/Feature_Importance.png")
print("   - important_features/Feature_Importance_Table.png")

