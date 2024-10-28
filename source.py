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


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[7]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

