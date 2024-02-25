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



industries = {
    "Mining, logging, construction": [0, 0, 0, 0],
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
        "Mining, logging, construction": [101700	104700	83600	107100	127600	129900	109600],
        "Manufacturing": [212800	171300	167000	188200	184300	168400	142200],
        "Trade, transportation, and utilities": [325600	313200	301600	354400	398000	390300	332600],
        "Information": [79500	77700	87700	97500	128400	133700	139000],
        "Financial activities": [101800	106700	92100	95900	101400	100400	87600],
        "Professional and business services": [220500	214400	220700	268600	302100	295700	277500],
        "Education and health services": [183700	198400	231500	251300	283000	272100	223500],
        "Leisure and hospitality": [145800	152500	155700	185200	207800	150600	133000],
        "Other services": [57800	61800	63200	70200	78700	71100	59300],
        "Government": [236000	252100	264200	270300	275500	266000	206700],
    },
        'Omaha': {
            "Mining, logging, construction": [23500	25700	20900	25800	30500	30400	30700],
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
        'Scranton': {
            "Mining, logging, construction": [0, 0, 0, 0],
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
}