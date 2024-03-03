import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score


# creates a filled in numpy array of years from 2011 to 2021
censusYears = np.arange(2011, 2022, 1)

# Manchester Census Data
manchesterAgeCensus2011 = {
    "00-04": 36413,
    "05-09": 28617,
    "10-14": 26785,
    "15-19": 38302,
    "20-24": 66998,
    "25-29": 56595,
    "30-34": 45004,
    "35-39": 33970,
    "40-44": 32446,
    "45-49": 28637,
    "50-54": 23927,
    "55-59": 19633,
    "60-64": 18236,
    "65-69": 13191,
    "70-74": 11576,
    "75-79": 9378,
    "80-84": 7026,
    "85-89": 4226,
    "90+": 2147,
}


manchesterAgeCensus2021 = {
    "00-04": 34400,
    "05-09": 36600,
    "10-14": 36300,
    "15-19": 42400,
    "20-24": 61900,
    "25-29": 52600,
    "30-34": 48000,
    "35-39": 42000,
    "40-44": 35800,
    "45-49": 31100,
    "50-54": 30400,
    "55-59": 26700,
    "60-64": 21600,
    "65-69": 16400,
    "70-74": 13800,
    "75-79": 9200,
    "80-84": 6700,
    "85-89": 3900,
    "90+": 2100,
}


# Housing Data for Manchester
manchesterHousingData = {
    "Bungalow": np.array([
        1780,
        1840,
        1950,
        1990,
        2070,
        2140,
        2250,
        2310,
        2380,
        2420,
        2470,
        2500,
        2570,
        2620,
        2620,
        2670,
        2680,
        2690,
        2710,
        2760,
        2770,
        2780,
        2830,
        2830,
        2840,
        2840,
        2840,
        2850,
        2850,
    ]),
    "Flat/Maisonette": np.array([
        39620,
        40270,
        42600,
        44750,
        47210,
        48770,
        50210,
        51860,
        54060,
        56450,
        59080,
        61170,
        63280,
        65170,
        69390,
        74130,
        76120,
        77790,
        78420,
        78940,
        80210,
        80300,
        80600,
        81500,
        82470,
        83760,
        85420,
        88120,
        92150,

    ]),
    "Terraced House": np.array([
        59010,
        60770,
        62650,
        64670,
        66660,
        69360,
        71750,
        74100,
        76170,
        77070,
        77500,
        77840,
        77710,
        77680,
        78140,
        78600,
        78670,
        78550,
        78180,
        78390,
        78650,
        78740,
        78930,
        79340,
        79640,
        79860,
        80050,
        80260,
        80430,
    ]),
    "Semi-Detached House": np.array([
        39060,
        40350,
        41790,
        43140,
        44590,
        46310,
        47930,
        49360,
        50820,
        51500,
        51810,
        52020,
        52260,
        52340,
        52550,
        52790,
        52890
        53010,
        53090,
        53280,
        53590,
        53850,
        54100,
        54540,
        54880,
        55150,
        55630,
        55990,
        56280,
    ])
    "Detached House": np.array([
        2710,
        2860,
        3070,
        3290,
        3490,
        3730,
        3970,
        4210,
        4410,
        4540,
        4610,
        4740,
        4880,
        4990,
        5170,
        5320,
        5370,
        5440,
        5480,
        5530,
        5640,
        5730,
        5820,
        5910,
        5990,
        6080,
        6180,
        6240,
        6290,
    ])
     "Annexe": np.array([
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        10,
        10,
        10,
        10,
        10,
        20,
        20,
        20,
        20,
        20,
        20,
        30,
        30,
        30,
        30,
        30,

     ])
     "Mobile Home": np.array([
        80,
        80,
        80,
        90,
        90,
        80,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
        90,
        110,
        110,
        110,
        110,
        110,
        120,
        120,
        120,
        120,

     ])
      "Unknown": np.array([
        44810,
        41280,
        34400,
        30170,
        25090,
        19120,
        14680,
        10650,
        5890,
        2870,
        1450,
        800,
        400,
        350,
        330,
        350,
        330,
        300,
        290,
        280,
        350,
        330,
        320,
        320,
        340,
        460,
        650,
        670,
        670,

      ])

}




# Census Modelling For Number of People in Each Age Category In Manchester

# Gets the items in the census dictionary as a list then loops through it and plots 2011 vs 2021

# 2021 bar chart
plt.bar(manchesterAgeCensus2021.keys(), manchesterAgeCensus2021.values())
plt.title("2021 Census Data On Age Distribution In Manchester")
# rotates the x axis labels by 90 degrees
plt.xticks(rotation=90)
# labels the x axis
plt.xlabel("Age Categories")
# labels the y axis
plt.ylabel("Number of People")
plt.show()

# 2011 bar chart
plt.bar(manchesterAgeCensus2011.keys(), manchesterAgeCensus2011.values())
plt.title("2011 Census Data On Age Distribution In Manchester")
# rotates the x axis labels by 90 degrees
plt.xticks(rotation=90)
# labels the x axis
plt.xlabel("Age Categories")
# labels the y axis
plt.ylabel("Number of People")
plt.show()

for key, value in manchesterAgeCensus2021.items():
    pass




# Brigton and Hove Census Data
brightonAgeCensus2011 = {
    "00-04": 15015,
    "05-09": 13291,
    "10-14": 13412,
    "15-19": 18039,
    "20-24": 28129,
    "25-29": 22998,
    "30-34": 21959,
    "35-39": 21789,
    "40-44": 21905,
    "45-49": 20443,
    "50-54": 15345,
    "55-59": 12638,
    "60-64": 12714,
    "65-69": 9535,
    "70-74": 7925,
    "75-79": 6676,
    "80+":11556
}

brightonAgeCensus2021 = {
    "00-04": 11700,
    "05-09": 13000,
    "10-14": 14200,
    "15-19": 17800,
    "20-24": 28000,
    "25-29": 21000,
    "30-34": 20500,
    "35-39": 19400,
    "40-44": 19300,
    "45-49": 20400,
    "50-54": 20900,
    "55-59": 18400,
    "60-64": 13600,
    "65-69": 10900,
    "70-74": 10300,
    "75-79": 7100,
    "80+": 10800,
}