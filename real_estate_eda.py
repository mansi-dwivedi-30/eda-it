#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exploratory Data Analysis (EDA) for Real Estate Pricing
=======================================================
This script performs a comprehensive analysis of housing data to identify 
factors that influence house prices in the real estate market.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set the style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv('housing_data.csv')

# Display basic information about the dataset
print("\n=== Dataset Information ===")
print(f"Shape of the dataset: {df.shape}")
print(f"Number of features: {df.shape[1]}")
print(f"Number of samples: {df.shape[0]}")

# Display the first few rows of the dataset
print("\n=== First 5 rows of the dataset ===")
print(df.head())

# Data Cleaning
print("\n=== Data Cleaning ===")

# Check for missing values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
print("Features with missing values:")
print(missing_data[missing_data['Missing Values'] > 0].sort_values('Missing Values', ascending=False))

# Handle missing values for important features
# For numerical features, fill with median
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
for feature in numerical_features:
    if df[feature].isnull().sum() > 0:
        df[feature].fillna(df[feature].median(), inplace=True)

# For categorical features, fill with mode
categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    if df[feature].isnull().sum() > 0:
        df[feature].fillna(df[feature].mode()[0], inplace=True)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Feature Engineering
print("\n=== Feature Engineering ===")

# Create new features
# 1. Property Age (as of 2025)
df['PropertyAge'] = 2025 - df['YearBuilt']

# 2. Price per square foot
df['PricePerSqFt'] = df['SalePrice'] / df['GrLivArea']

# 3. Total Bathrooms
df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])

# 4. Total Square Footage
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']

# 5. Has Pool
df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

# 6. Has Garage
df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

# 7. Has Fireplace
df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# 8. Remodeled
df['Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)

# 9. Quality Score (combining overall quality and condition)
df['QualityScore'] = df['OverallQual'] * df['OverallCond']

# 10. Total Porch Area
porch_columns = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
df['TotalPorchArea'] = df[porch_columns].sum(axis=1)

print("New features created:")
print(df[['PropertyAge', 'PricePerSqFt', 'TotalBathrooms', 'TotalSF', 'HasPool', 
          'HasGarage', 'HasFireplace', 'Remodeled', 'QualityScore', 'TotalPorchArea']].head())

# Univariate Analysis
print("\n=== Univariate Analysis ===")

# Function to save plots
def save_plot(plot_name):
    plt.tight_layout()
    plt.savefig(f'plots/{plot_name}.png')
    plt.close()

# Create plots directory if it doesn't exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Distribution of Sale Price
plt.figure(figsize=(12, 6))
sns.histplot(df['SalePrice'], kde=True)
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price ($)')
plt.ylabel('Frequency')
save_plot('sale_price_distribution')

# Log-transformed Sale Price for better visualization
plt.figure(figsize=(12, 6))
sns.histplot(np.log1p(df['SalePrice']), kde=True)
plt.title('Distribution of Log-Transformed Sale Price')
plt.xlabel('Log(Sale Price)')
plt.ylabel('Frequency')
save_plot('log_sale_price_distribution')

# Distribution of key numerical features
numerical_cols = ['GrLivArea', 'TotalBathrooms', 'BedroomAbvGr', 'PropertyAge', 'TotalSF']
fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(15, 4*len(numerical_cols)))

for i, feature in enumerate(numerical_cols):
    # Histogram
    sns.histplot(df[feature], kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Distribution of {feature}')
    
    # Box plot
    sns.boxplot(x=df[feature], ax=axes[i, 1])
    axes[i, 1].set_title(f'Box Plot of {feature}')

save_plot('numerical_features_distribution')

# Distribution of key categorical features
categorical_cols = ['OverallQual', 'Neighborhood', 'HouseStyle', 'SaleCondition']
fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(15, 5*len(categorical_cols)))

for i, feature in enumerate(categorical_cols):
    # Count plot
    sns.countplot(y=df[feature], order=df[feature].value_counts().index[:15], ax=axes[i])
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].set_xlabel('Count')
    axes[i].set_ylabel(feature)

save_plot('categorical_features_distribution')

# Multivariate Analysis
print("\n=== Multivariate Analysis ===")

# Correlation matrix of numerical features
correlation_cols = ['SalePrice', 'GrLivArea', 'TotalBathrooms', 'BedroomAbvGr', 
                    'PropertyAge', 'TotalSF', 'OverallQual', 'QualityScore', 
                    'PricePerSqFt', 'TotalPorchArea', 'GarageArea']

correlation_matrix = df[correlation_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Key Features')
save_plot('correlation_matrix')

# Top 10 features with highest correlation with SalePrice
top_corr = correlation_matrix['SalePrice'].sort_values(ascending=False)[1:11]
print("Top 10 features correlated with Sale Price:")
print(top_corr)

# Scatter plots of key features vs Sale Price
key_features = ['GrLivArea', 'TotalBathrooms', 'OverallQual', 'TotalSF', 'QualityScore']
fig, axes = plt.subplots(len(key_features), 1, figsize=(12, 4*len(key_features)))

for i, feature in enumerate(key_features):
    sns.scatterplot(x=df[feature], y=df['SalePrice'], ax=axes[i])
    axes[i].set_title(f'{feature} vs Sale Price')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Sale Price ($)')

save_plot('key_features_vs_sale_price')

# Size Impact Analysis
print("\n=== Size Impact Analysis ===")

# Relationship between bedrooms, bathrooms, and price
plt.figure(figsize=(12, 8))
sns.scatterplot(x='BedroomAbvGr', y='SalePrice', size='TotalBathrooms', 
                hue='OverallQual', data=df, sizes=(50, 200))
plt.title('Relationship between Bedrooms, Bathrooms, and Price')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Sale Price ($)')
save_plot('bedrooms_bathrooms_price')

# Price per square foot by neighborhood
plt.figure(figsize=(14, 8))
neighborhood_price = df.groupby('Neighborhood')['PricePerSqFt'].median().sort_values(ascending=False)
sns.barplot(x=neighborhood_price.index[:15], y=neighborhood_price.values[:15])
plt.title('Median Price per Square Foot by Neighborhood (Top 15)')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Neighborhood')
plt.ylabel('Median Price per Square Foot ($)')
save_plot('price_per_sqft_by_neighborhood')

# Market Trends and Historical Pricing
print("\n=== Market Trends and Historical Pricing ===")

# Sale price trends over the years
yearly_price = df.groupby('YrSold')['SalePrice'].median()
plt.figure(figsize=(12, 6))
sns.lineplot(x=yearly_price.index, y=yearly_price.values, marker='o', linewidth=2)
plt.title('Median Sale Price Trend Over Years')
plt.xlabel('Year Sold')
plt.ylabel('Median Sale Price ($)')
save_plot('price_trend_over_years')

# Price trends by month
monthly_price = df.groupby('MoSold')['SalePrice'].median()
plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_price.index, y=monthly_price.values, marker='o', linewidth=2)
plt.title('Median Sale Price by Month')
plt.xlabel('Month Sold')
plt.ylabel('Median Sale Price ($)')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
save_plot('price_trend_by_month')

# Price distribution by sale condition
plt.figure(figsize=(12, 6))
sns.boxplot(x='SaleCondition', y='SalePrice', data=df)
plt.title('Price Distribution by Sale Condition')
plt.xlabel('Sale Condition')
plt.ylabel('Sale Price ($)')
plt.xticks(rotation=45)
save_plot('price_by_sale_condition')

# Customer Preferences and Amenities
print("\n=== Customer Preferences and Amenities ===")

# Impact of amenities on price
amenities = ['HasGarage', 'HasFireplace', 'HasPool', 'Remodeled']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, amenity in enumerate(amenities):
    sns.boxplot(x=df[amenity], y=df['SalePrice'], ax=axes[i])
    axes[i].set_title(f'Impact of {amenity} on Sale Price')
    axes[i].set_xlabel(f'Has {amenity.replace("Has", "")}' if 'Has' in amenity else amenity)
    axes[i].set_ylabel('Sale Price ($)')
    axes[i].set_xticklabels(['No', 'Yes'])

save_plot('amenities_impact')

# Property quality vs price
plt.figure(figsize=(12, 8))
sns.boxplot(x='OverallQual', y='SalePrice', data=df)
plt.title('Sale Price by Overall Quality')
plt.xlabel('Overall Quality (1-10)')
plt.ylabel('Sale Price ($)')
save_plot('quality_vs_price')

# Clustering Analysis
print("\n=== Clustering Analysis ===")

# Select features for clustering
cluster_features = ['GrLivArea', 'TotalBathrooms', 'OverallQual', 'PropertyAge', 'TotalSF']
X = df[cluster_features].copy()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using elbow method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(k_range)
save_plot('elbow_method')

# Apply K-means clustering with the optimal number of clusters (let's use 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_stats = df.groupby('Cluster').agg({
    'SalePrice': ['mean', 'median', 'count'],
    'GrLivArea': 'mean',
    'TotalBathrooms': 'mean',
    'OverallQual': 'mean',
    'PropertyAge': 'mean',
    'TotalSF': 'mean'
})

print("\nCluster Statistics:")
print(cluster_stats)

# Visualize clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='Cluster', data=df, palette='viridis', s=80)
plt.title('Clusters by Living Area and Sale Price')
plt.xlabel('Above Ground Living Area (sq ft)')
plt.ylabel('Sale Price ($)')
save_plot('clusters_visualization')

# Summary of findings
print("\n=== Summary of Findings ===")
print("1. Key factors influencing house prices include:")
print("   - Living area (square footage)")
print("   - Overall quality and condition")
print("   - Number of bathrooms")
print("   - Location (neighborhood)")
print("   - Presence of amenities like garage and fireplace")

print("\n2. Price per square foot varies significantly by neighborhood")
print("3. Property age has a negative correlation with price")
print("4. Sale prices show seasonal variations with peaks in certain months")
print("5. Clustering analysis revealed distinct property segments in the market")

print("\nEDA completed successfully. Check the 'plots' directory for visualizations.")
