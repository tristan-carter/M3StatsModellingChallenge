import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
percentageByAge={
    20:12,
    40:29,
    60:25,
    62:22
}
percentageByIncome={
    7500:8,
    17500:24,
    25000:21,
    35000:32,
    45000:38
}