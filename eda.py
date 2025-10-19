import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plot_folder = 'important_feature_plots'
os.makedirs(plot_folder, exist_ok=True)

df = pd.read_csv('crop.csv') 

df.columns = df.columns.str.strip()

print("Columns in dataset:", df.columns)

target = [col for col in df.columns if 'yield' in col.lower()][0]

categorical_features = df.select_dtypes(include='object').columns.tolist()
categorical_features = [col for col in categorical_features if col != target]

numerical_features = df.select_dtypes(include=['int64','float64']).columns.tolist()
numerical_features = [col for col in numerical_features if col != target]

print("\nTarget column:", target)
print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nSummary statistics for numerical features:")
print(df[numerical_features].describe())

corr_matrix = df[numerical_features + [target]].corr()
print("\nCorrelation with Yield:")
print(corr_matrix[target].sort_values(ascending=False))

for feature in categorical_features:
    print(f"\nAverage Yield by {feature}:")
    print(df.groupby(feature)[target].mean().sort_values(ascending=False))

corr_with_yield = corr_matrix[target].abs().sort_values(ascending=False)
important_numerical = corr_with_yield[1:6].index.tolist()  
print("\nImportant numerical features:", important_numerical)

important_categorical = []
for feature in categorical_features:
    avg_yield = df.groupby(feature)[target].mean()
    diff = avg_yield.max() - avg_yield.min()
    if diff > df[target].std(): 
        important_categorical.append(feature)
print("Important categorical features:", important_categorical)

important_features = important_numerical + important_categorical
print("\nOverall important features for Yield prediction:", important_features)

for feature in important_numerical:
    plt.figure(figsize=(8,5))
    sns.histplot(df[feature], bins=20, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.savefig(f'{plot_folder}/hist_{feature}.png')
    plt.close()

for feature in important_categorical:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=feature, y=target, data=df)
    plt.xticks(rotation=45)
    plt.title(f'{feature} vs {target}')
    plt.savefig(f'{plot_folder}/box_{feature}_vs_{target}.png')
    plt.close()

for feature in important_numerical:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=feature, y=target, data=df)
    plt.title(f'{feature} vs {target}')
    plt.savefig(f'{plot_folder}/scatter_{feature}_vs_{target}.png')
    plt.close()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix.loc[important_numerical + [target], important_numerical + [target]],
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Important Numerical Features)')
plt.savefig(f'{plot_folder}/correlation_heatmap.png')
plt.close()

print(f"All plots for important features saved in folder: {plot_folder}")
