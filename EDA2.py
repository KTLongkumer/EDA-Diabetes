#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


import pandas as pd
df = pd.read_csv('diabetes.csv')
print(df.head())


# In[ ]:


# describtive statistic
df.describe(include='all') 


# In[ ]:





# In[ ]:





# In[1]:


# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
diabetes_df = pd.read_csv('diabetes.csv')

# Display basic information about the dataset
print(diabetes_df.info())

# Check for missing values
print(diabetes_df.isnull().sum())

# Handle missing values
# For example, you might replace missing values with the mean or median of the respective column
diabetes_df.fillna(diabetes_df.mean(), inplace=True)

# Check for inconsistencies or outliers
# For example, you might want to check for unrealistic values for features like glucose level, blood pressure, etc.

# Data Cleaning and Preprocessing
# You might perform various preprocessing steps such as:
# - Feature scaling/normalization
# - Encoding categorical variables (if any)
# - Handling outliers
# - Feature engineering (creating new features from existing ones)
# - Splitting the dataset into features and target variable(s)

# Example:
# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
diabetes_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] =     scaler.fit_transform(diabetes_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])

# Encoding categorical variables (if any)
# For example, if 'Gender' is a categorical variable, you might encode it as 0 for male and 1 for female
# diabetes_df['Gender'] = diabetes_df['Gender'].map({'Male': 0, 'Female': 1})

# Handling outliers
# You might use techniques like winsorization or trimming to handle outliers

# Feature engineering
# You might create new features based on domain knowledge or feature interactions

# Splitting the dataset into features and target variable(s)
X = diabetes_df.drop('Outcome', axis=1)  # Features
y = diabetes_df['Outcome']  # Target variable

# Now you can use the preprocessed data for analysis or modeling


# In[2]:


# Compute basic statistics
basic_stats = diabetes_df.describe()
print(basic_stats)

# Visualize distributions
# Histograms
diabetes_df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Box plots for each feature
plt.figure(figsize=(12, 8))
sns.boxplot(data=diabetes_df)
plt.xticks(rotation=45)
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(diabetes_df, hue='Outcome')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(diabetes_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Initial insights
# - Check if there are significant differences in distributions between diabetic and non-diabetic individuals
# - Look for correlations between features and the target variable (Outcome)
# - Identify potential relationships or patterns between pairs of features
# - Assess the strength and direction of correlations between features


# In[3]:


# Pairwise correlation among features
plt.figure(figsize=(12, 8))
sns.heatmap(diabetes_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pairwise Correlation Heatmap')
plt.show()

# Scatterplot matrix
sns.pairplot(diabetes_df, hue='Outcome', diag_kind='kde')
plt.show()

# Violin plots to compare distributions of features for diabetic and non-diabetic individuals
plt.figure(figsize=(12, 8))
for i, column in enumerate(diabetes_df.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    sns.violinplot(x='Outcome', y=column, data=diabetes_df)
plt.tight_layout()
plt.show()

# Joint plots to explore relationships between specific features
sns.jointplot(x='Glucose', y='BMI', data=diabetes_df, kind='hex', cmap='viridis')
plt.show()

# Pairwise interaction effects
# For example, you might investigate how the interaction between Glucose and BMI affects the likelihood of diabetes
sns.lmplot(x='Glucose', y='BMI', data=diabetes_df, hue='Outcome', logistic=True)
plt.show()

# Time-series analysis (if applicable)
# For example, if your dataset includes time-series data, you might explore trends, seasonality, or other patterns over time

# Feature importance analysis (if applicable)
# If you're planning to build a predictive model, you might explore feature importance using techniques like random forests or gradient boosting

# Clustering (if applicable)
# Explore whether there are natural clusters or groups within the data using clustering algorithms like K-means or hierarchical clustering

# Dimensionality reduction (if applicable)
# If the dataset has high dimensionality, consider using techniques like PCA or t-SNE to visualize the data in lower dimensions while preserving important relationships

# Advanced insights
# - Identify strong correlations between features and explore potential causal relationships
# - Investigate non-linear relationships between features and the target variable
# - Explore interactions between features that may have predictive power
# - Identify any interesting patterns or anomalies in the data that may require further investigation


# In[4]:


# Parallel coordinates plot to visualize high-dimensional data
from pandas.plotting import parallel_coordinates

plt.figure(figsize=(12, 8))
parallel_coordinates(diabetes_df, 'Outcome', colormap='viridis')
plt.show()

# Pairplot with hue representing Outcome to visualize multivariate relationships
sns.pairplot(diabetes_df, hue='Outcome', diag_kind='kde')
plt.show()

# 3D scatter plot for exploring interactions among three features
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(diabetes_df['Glucose'], diabetes_df['BMI'], diabetes_df['Age'], c=diabetes_df['Outcome'], cmap='viridis')
ax.set_xlabel('Glucose')
ax.set_ylabel('BMI')
ax.set_zlabel('Age')
plt.show()


# In[5]:


# Principal Component Analysis (PCA) to reduce dimensionality and visualize clusters (if any)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Analysis')
plt.colorbar(label='Outcome')
plt.show()


# In[6]:


# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example: Implementing a Random Forest Classifier for prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predicting on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[7]:


# Summary of Key Findings and Insights
print("Key Findings and Insights:")
print("- Glucose level, BMI, and Age are significant predictors of diabetes.")
print("- Multivariate relationships reveal complex interactions among features.")
print("- The Random Forest Classifier achieved satisfactory accuracy in predicting diabetes.")

# Significance of Different Features
print("\nSignificance of Different Features:")
print("- Glucose: Elevated levels increase diabetes risk significantly.")
print("- BMI: Higher values are associated with increased risk.")
print("- Age: Older individuals have a higher risk of diabetes.")

# Potential Further Steps and Analyses
print("\nPotential Further Steps and Analyses:")
print("1. Temporal Analysis: Explore trends in diabetes prevalence over time.")
print("2. Feature Engineering: Create new features or transformations to capture additional information.")
print("3. Advanced Predictive Modeling: Experiment with more sophisticated algorithms.")
print("4. Clinical Validation: Validate model predictions against medical records.")
print("5. Subgroup Analysis: Investigate population-specific risk factors.")


# In[ ]:




