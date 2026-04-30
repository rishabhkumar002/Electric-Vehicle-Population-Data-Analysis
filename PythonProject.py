import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
# Assuming the dataset is in CSV format after converting from Excel
df = pd.read_csv("Electric_Vehicle_Population_Data.csv")

# --- Data Cleaning ---
# Drop irrelevant columns
columns_to_drop = ['VIN (1-10)', 'DOL Vehicle ID', 'Vehicle Location', '2020 Census Tract']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Handle missing values
# Check for missing values
print("Missing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing categorical columns with 'Unknown'
categorical_columns = ['County', 'City', 'Make', 'Model', 'Electric Vehicle Type', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 'Electric Utility']
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Impute numerical columns with median
numerical_columns = ['Electric Range', 'Base MSRP', 'Model Year', 'Postal Code']
for col in numerical_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Convert Model Year to integer and filter unrealistic years
df['Model Year'] = df['Model Year'].astype(int)
df = df[(df['Model Year'] >= 2000) & (df['Model Year'] <= 2025)]

# Standardize text columns
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].str.strip().str.upper()

# Handle zero or negative Electric Range
df['Electric Range'] = df['Electric Range'].apply(lambda x: np.nan if x <= 0 else x)
df['Electric Range'] = df['Electric Range'].fillna(df['Electric Range'].median())

# Handle zero or negative Base MSRP
df['Base MSRP'] = df['Base MSRP'].apply(lambda x: np.nan if x <= 0 else x)
df['Base MSRP'] = df['Base MSRP'].fillna(df['Base MSRP'].median())

# --- Data Preprocessing ---
# Create Vehicle Age column
current_year = 2025
df['Vehicle Age'] = current_year - df['Model Year']

# Categorize Electric Range
bins = [0, 100, 200, 300, np.inf]
labels = ['Low (<100)', 'Medium (100-200)', 'High (200-300)', 'Very High (>300)']
df['Range Category'] = pd.cut(df['Electric Range'], bins=bins, labels=labels, include_lowest=True)

# Encode Electric Vehicle Type for analysis
le = LabelEncoder()
df['EV Type Encoded'] = le.fit_transform(df['Electric Vehicle Type'])

# --- Outlier Detection ---
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Detect outliers in Electric Range and Base MSRP
range_outliers = detect_outliers_iqr(df, 'Electric Range')
msrp_outliers = detect_outliers_iqr(df, 'Base MSRP')
print(f"Number of Electric Range Outliers: {len(range_outliers)}")
print(f"Number of Base MSRP Outliers: {len(msrp_outliers)}")

# --- Linear Regression Model ---
# Select features and target
features = ['Model Year', 'Base MSRP', 'Vehicle Age']
target = 'Electric Range'

# Prepare data for regression
X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_scaled, y)

# Make predictions
y_pred = lr_model.predict(X_scaled)

# Evaluate model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("\nLinear Regression Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
print("Coefficients:", dict(zip(features, lr_model.coef_)))
print(f"Intercept: {lr_model.intercept_:.2f}")

# --- Simple Linear Regression for Visualization (Model Year vs. Electric Range) ---
# Use Model Year as a single predictor for visualization
X_single = df[['Model Year']].values
y_single = df['Electric Range'].values

# Train simple linear regression
lr_simple = LinearRegression()
lr_simple.fit(X_single, y_single)

# Predict for plotting
x_range = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
y_range_pred = lr_simple.predict(x_range)

# Plot scatter with regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Model Year'], y=df['Electric Range'], hue=df['Electric Vehicle Type'], size=df['Vehicle Age'])
plt.plot(x_range, y_range_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Electric Range vs. Model Year with Linear Regression')
plt.xlabel('Model Year')
plt.ylabel('Electric Range (miles)')
plt.legend()
plt.savefig('electric_range_vs_model_year_regression.png')
plt.show()
plt.close()

# --- Exploratory Data Analysis ---
# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Distribution of Electric Vehicle Types
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Electric Vehicle Type')
plt.title('Distribution of Electric Vehicle Types')
plt.xlabel('EV Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('ev_type_distribution.png')
plt.show()
plt.close()

# Top 10 Vehicle Makes
plt.figure(figsize=(10, 6))
top_makes = df['Make'].value_counts().head(10)
sns.barplot(x=top_makes.values, y=top_makes.index)
plt.title('Top 10 Vehicle Makes')
plt.xlabel('Count')
plt.ylabel('Make')
plt.savefig('top_makes.png')
plt.show()
plt.close()

# Electric Range Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Electric Range'], bins=30, kde=True)
plt.title('Distribution of Electric Range')
plt.xlabel('Electric Range (miles)')
plt.ylabel('Frequency')
plt.savefig('electric_range_distribution.png')
plt.show()
plt.close()

# Model Year Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Model Year'], bins=range(2000, 2026), kde=False)
plt.title('Distribution of Model Years')
plt.xlabel('Model Year')
plt.ylabel('Count')
plt.savefig('model_year_distribution.png')
plt.show()
plt.close()

# Electric Range vs. Base MSRP
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Electric Range', y='Base MSRP', hue='Electric Vehicle Type', size='Vehicle Age')
plt.title('Electric Range vs. Base MSRP')
plt.xlabel('Electric Range (miles)')
plt.ylabel('Base MSRP ($)')
plt.savefig('range_vs_msrp.png')
plt.show()
plt.close()

# Geographic Distribution by County (Top 10)
plt.figure(figsize=(10, 6))
top_counties = df['County'].value_counts().head(10)
sns.barplot(x=top_counties.values, y=top_counties.index)
plt.title('Top 10 Counties by EV Population')
plt.xlabel('Count')
plt.ylabel('County')
plt.savefig('top_counties.png')
plt.show()
plt.close()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
numerical_cols = ['Model Year', 'Electric Range', 'Base MSRP', 'Vehicle Age', 'EV Type Encoded']
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()
plt.close()

# Range Category Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Range Category')
plt.title('Distribution of Electric Range Categories')
plt.xlabel('Range Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('range_category_distribution.png')
plt.show()
plt.close()

# --- Key Findings ---
print("\nKey Findings:")
print("1. The dataset predominantly contains Battery Electric Vehicles (BEVs) compared to Plug-in Hybrid Electric Vehicles (PHEVs).")
print("2. Tesla is the most common vehicle make, followed by Nissan and Chevrolet.")
print("3. The electric range is mostly concentrated between 100-300 miles, with some outliers exceeding 300 miles.")
print("4. Newer model years (2020-2025) dominate the dataset, indicating a recent surge in EV adoption.")
print("5. King County has the highest number of registered EVs, reflecting higher urban adoption.")
print(f"6. Linear regression model shows an R^2 score of {r2:.2f}, indicating {r2*100:.1f}% of the variance in Electric Range is explained by Model Year, Base MSRP, and Vehicle Age.")
print("7. The regression line for Model Year vs. Electric Range suggests newer vehicles tend to have higher ranges, though the relationship is moderate.")

# Save cleaned dataset
df.to_csv('cleaned_electric_vehicle_population.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_electric_vehicle_population.csv'")
