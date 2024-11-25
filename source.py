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


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[6]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

