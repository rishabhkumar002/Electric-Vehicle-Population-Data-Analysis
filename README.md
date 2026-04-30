# 🚀 EV Data Analysis & Range Prediction using Machine Learning

## 📌 Project Overview
This project focuses on analyzing Electric Vehicle (EV) population data to extract meaningful insights and build a machine learning model to predict electric range.

The goal is to understand EV adoption trends, key influencing factors, and performance patterns.

---

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn

---

## 🧠 What This Project Does
✔ Cleans and preprocesses real-world EV dataset  
✔ Handles missing values and outliers  
✔ Performs Exploratory Data Analysis (EDA)  
✔ Creates meaningful visualizations  
✔ Builds a Linear Regression model to predict electric range  

---

## 🛠️ Key Steps

### 1. Data Cleaning
- Removed irrelevant columns
- Handled missing values using median and "Unknown"
- Standardized categorical data
- Removed invalid values (0 or negative range/MSRP)

### 2. Feature Engineering
- Created **Vehicle Age** feature
- Categorized Electric Range into segments
- Encoded EV types for modeling

### 3. Outlier Detection
- Used IQR method to identify anomalies in:
  - Electric Range
  - Base MSRP

### 4. Machine Learning Model
- Model Used: **Linear Regression**
- Features:
  - Model Year
  - Base MSRP
  - Vehicle Age

### 5. Visualization (EDA)
- EV type distribution
- Top vehicle makes
- Electric range distribution
- Model year trends
- Range vs Price analysis
- Geographic distribution (Top counties)
- Correlation heatmap

---

## 📊 Results & Insights
- Majority of vehicles are **Battery Electric Vehicles (BEVs)**
- **Tesla** dominates the dataset
- Electric range mostly falls between **100–300 miles**
- EV adoption increased significantly after **2020**
- **King County** has the highest EV population
- Model performance:
  - R² Score: ~0.11 (moderate relationship)

---

