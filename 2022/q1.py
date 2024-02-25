import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

# data
years = np.array([2002,2003,2004,2005,2006,2007,2008,2009,2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
percentWorkersHome = np.array([
    2.90,
    3.10,
    3.00,
    3.10,
    3.20,
    3.30,
    3.50,
    3.70,
    3.50,
    4.00,
    4.10,
    4.30,
    4.50,
    4.40,
    4.90,
    4.80,
    5.10,
    5.30,
])

# plot with linear regression
plt.scatter(years, percentWorkersHome)
plt.xlabel('Year')
plt.ylabel('Percent of Workers Working from Home')
plt.title('Percent of Workers Working from Home by Year')

# adjusts x ticks
plt.xticks(np.arange(2002, 2020, 2))

# plots linear regression
m, b = np.polyfit(years, percentWorkersHome, 1)
plt.plot(years, m*years + b)

# calculates r^2
r2 = r2_score(percentWorkersHome, m*years + b)
print('r^2:', r2)

# extrapolates linear regression to 2027
#extrapolatedYears = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027])
#extrapolatedPercentWorkersHome = m*extrapolatedYears + b
#plt.plot(extrapolatedYears, extrapolatedPercentWorkersHome, 'r--')

plt.show()



# import data from excel spreadsheet
df = pd.read_excel(r"C:\Users\ticar\Downloads\Remote-work-data.xlsx")

industries = {
    "Mining, logging, construction": 0,
    "Manufacturing": 0,
    "Trade, transportation, and utilities": 0,
    "Information": 0,
    "Financial activities": 0,
    "Professional and business services": 0,
    "Education and health services": 0,
    "Leisure and hospitality": 0,
    "Other services": 0,
    "Government": 0,
},

numUKEmployeesByIndustryByCity = {
    "Seattle": {
        "Mining, logging, construction": 0,
        "Manufacturing": 0,
        "Trade, transportation, and utilities": 0,
        "Information": 0,
        "Financial activities": 0,
        "Professional and business services": 0,
        "Education and health services": 0,
        "Leisure and hospitality": 0,
        "Other services": 0,
        "Government": 0,
    },
    'Omaha': industries,
    'Scranton': industries,
}

# fills in numUKEmployeesByIndustryByCity



numUSEmployeesByIndustryByCity = {
    'Liverpool': industries,
    'Barry': industries,
}