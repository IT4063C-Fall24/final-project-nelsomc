#!/usr/bin/env python
# coding: utf-8

# # {Project Title}📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# Some college football teams/programs are known for winning all the time. What is the difference between these teams and teams that do not.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# How do college football dynasties happen? What gives a college football team the edge to win? Are there certain factors that great teams have that help them win?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# The answers could be anywhere from ability to recruit, if it was just straight ability to coach, could it be location and weather, and so on. As for graphics it will be charts that can describe different aspects of the game and give percentages and numbers back showing different things including win probability.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# I found a great website that allows you to query for data and download csv files. I have also found a dataset on Kaggle. I am also going to use Pro Football Focus' (PFF) data.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# 
# Working for the team here at UC gives me excellent resources and I am going to start with finding data I think I will need. I have a meeting scheduled with the guy who is in charge of the teams analytics about what he does and where he finds his data and how he uses it.

# In[1]:


# Start your code here
import opendatasets as od
import pandas as pd

od.download('https://www.kaggle.com/datasets/jeffgallini/college-football-team-stats-2019', './data')


# In[5]:


import pandas as pd


df = pd.read_csv('Data/Data1.csv')



print(df.head())


# In[6]:


import pandas as pd


df = pd.read_csv('Data/Recruiting.csv')



print(df.head())


# ## Explatory Data Analysis
# 

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


data1 = pd.read_csv("./Data/Data1.csv")
recruiting_data = pd.read_csv("./Data/Recruiting.csv")

merged_data = pd.merge(data1, recruiting_data, on=["Year", "Team"], how="left")

data1_summary = data1.describe()
recruiting_data_summary = recruiting_data.describe()

plt.figure(figsize=(8, 5))
sns.histplot(data=data1, x="Total Wins", bins=10, kde=True)
plt.title("Distribution of Total Wins")
plt.xlabel("Total Wins")
plt.ylabel("Frequency")
plt.show()

numeric_data1 = data1.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data1.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap - Data1.csv")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=recruiting_data, x="Points")
plt.title("Boxplot of Points")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=merged_data, x="Points", y="Total Wins")
plt.title("Scatter Plot: Points vs Total Wins")
plt.xlabel("Points")
plt.ylabel("Total Wins")
plt.show()

data1_missing = data1.isnull().sum()
recruiting_data_missing = recruiting_data.isnull().sum()
data1_duplicates = data1.duplicated().sum()
recruiting_data_duplicates = recruiting_data.duplicated().sum()

{
    "Data1 Summary": data1_summary,
    "Recruiting Data Summary": recruiting_data_summary,
    "Missing Values in Data1": data1_missing,
    "Missing Values in Recruiting Data": recruiting_data_missing,
    "Duplicate Values in Data1": data1_duplicates,
    "Duplicate Values in Recruiting Data": recruiting_data_duplicates
}


# ## Data Cleaning

# In[5]:


data1_cleaned = data1.drop(columns=['Division'])

missing_values_data1 = data1_cleaned.isnull().sum()

data1_cleaned_info = data1_cleaned.info()

upper_limit = data1_cleaned['Total Wins'].quantile(0.95)
data1_cleaned = data1_cleaned[data1_cleaned['Total Wins'] <= upper_limit]

data1_cleaned = data1_cleaned.drop_duplicates()

merged_cleaned_data = pd.merge(data1_cleaned, recruiting_data, on=["Year", "Team"], how="left")

merged_cleaned_missing = merged_cleaned_data.isnull().sum()
merged_cleaned_shape = merged_cleaned_data.shape

{
    "Missing Values in Cleaned Data1": missing_values_data1,
    "Data1 Cleaned Info": data1_cleaned_info,
    "Outliers Removed (95th Percentile of Total Wins)": upper_limit,
    "Missing Values in Merged Dataset": merged_cleaned_missing,
    "Merged Dataset Shape": merged_cleaned_shape
}


# # **Machine Learning Plan**
# 
# ## **1. Type of Model**
# - We are using **Classification models** to predict whether a team will have a winning season (target: `win_flag`).
# - Models being tested:
#   - **Logistic Regression**: A baseline linear model for classification.
#   - **Random Forest Classifier**: A robust ensemble method for handling both numeric and categorical features.
# 
# ---
# 
# ## **2. Challenges Identified**
# - **Mixed Data Types**:
#   - The dataset contains both numeric (e.g., `ExpectedWins`, `total_wins`) and categorical (e.g., `team`, `Conference`) features.
#   - Categorical features need to be encoded before training.
# - **Imbalanced Data**:
#   - The distribution of the target variable (`win_flag`) is close to balanced but should be monitored.
# - **Missing Data**:
#   - Some features may have missing values that need to be handled appropriately.
# - **Feature Scaling**:
#   - Numeric features vary in scale and need normalization or standardization for optimal model performance.
# - **Dimensionality**:
#   - A mix of 22 features might require feature selection or dimensionality reduction to improve interpretability and avoid overfitting.
# 
# ---
# 
# ## **3. Plan to Address Challenges**
# - **Handling Missing Values**:
#   - Numeric features: Imputed using the mean.
#   - Categorical features: Imputed using the most frequent value.
# - **Encoding Categorical Variables**:
#   - One-hot encoding is applied to convert categorical features into numeric format.
# - **Scaling Numeric Features**:
#   - Applied StandardScaler to ensure numeric features are standardized (mean = 0, variance = 1).
# - **Model Evaluation**:
#   - Split data into training and testing sets (80/20 split).
#   - Evaluate performance using metrics such as accuracy, precision, recall, and F1-score.
# - **Feature Selection**:
#   - Used correlation analysis to identify and prioritize the most impactful features.
#   - Further feature importance analysis with Random Forest.
# - **Hyperparameter Tuning**:
#   - Experimented with parameters like `max_depth` for Random Forest and `C` for Logistic Regression to optimize performance.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[ ]:


data1_path = "./Data/Data1.csv"
recruiting_path = "./Data/Recruiting.csv"

data1 = pd.read_csv(data1_path)
recruiting = pd.read_csv(recruiting_path)

print("Columns in Data1:", data1.columns)
print("Columns in Recruiting:", recruiting.columns)

data1.rename(columns={'Team': 'team'}, inplace=True)
recruiting.rename(columns={'Team': 'team'}, inplace=True)

data1['team'] = data1['team'].str.strip().str.lower()
recruiting['team'] = recruiting['team'].str.strip().str.lower()

if 'year' in data1.columns and 'year' in recruiting.columns:
    df = pd.merge(data1, recruiting, how="inner", on=["team", "year"])
else:
    df = pd.merge(data1, recruiting, how="inner", on="team")

print("Merged DataFrame Info:")
print(df.info())
print("First 5 rows of the Merged DataFrame:")
print(df.head())


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

print("Dataset Summary:")
print(df.describe())

print("Missing Values:")
print(df.isnull().sum())

df.hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 12))

numeric_df = df.select_dtypes(include=['int64', 'float64'])

if numeric_df.empty:
    print("No numeric columns available for correlation heatmap.")
else:
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()


# In[71]:


if 'total_wins' in data1.columns and 'total_losses' in data1.columns:
    print("'total_wins' and 'total_losses' columns exist.")

    print("Checking for missing or invalid values:")
    print(data1[['total_wins', 'total_losses']].info())
    print(data1[['total_wins', 'total_losses']].head())

    data1['total_wins'] = pd.to_numeric(data1['total_wins'], errors='coerce')
    data1['total_losses'] = pd.to_numeric(data1['total_losses'], errors='coerce')

    print("Missing values after conversion:")
    print(data1[['total_wins', 'total_losses']].isnull().sum())


    data1['win_flag'] = (data1['total_wins'] > data1['total_losses']).astype(int)

    print("win_flag value counts:")
    print(data1['win_flag'].value_counts())
else:
    print("Columns 'total_wins' or 'total_losses' are missing.")


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(16,12))
sns.countplot(data1['win_flag'])
plt.title("Distribution of win_flag")
plt.xlabel("win_flag")
plt.ylabel("Count")
plt.show()



# In[ ]:


X = data1.drop(columns=['win_flag'], errors='ignore')
y = data1['win_flag']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# In[57]:


print("X_train sample:")
print(X_train.head())


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 30)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Overview of the Dataset
print("Dataset Overview:")
print(f"Shape: {data1.shape}")
print(data1.info())
print(data1.describe(include='all'))

# 2. Missing Values
print("\nMissing Values:")
missing_values = data1.isnull().sum()
print(missing_values[missing_values > 0])

# 3. Numeric Columns: Summary and Distribution
numeric_cols = data1.select_dtypes(include=['int64', 'float64']).columns
print("\nNumeric Columns Summary:")
print(data1[numeric_cols].describe())

# Visualize numeric distributions
data1[numeric_cols].hist(bins=15, figsize=(15, 10))
plt.suptitle("Numeric Column Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# 4. Categorical Columns: Frequency Counts
categorical_cols = data1.select_dtypes(include=['object']).columns
print("\nCategorical Columns Frequency Counts:")
for col in categorical_cols:
    print(f"\n{col} Value Counts:")
    print(data1[col].value_counts())


# 5. Correlation Heatmap (Numeric Columns)
if not numeric_cols.empty:
    plt.figure(figsize=(15, 10))
    sns.heatmap(data1[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title("Correlation Heatmap (Numeric Columns)", fontsize=16)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.show()
else:
    print("No numeric columns available for correlation heatmap.")


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[63]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

