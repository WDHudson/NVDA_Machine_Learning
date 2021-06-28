# Install dependencies
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')

# Store the data into a dataframe
df = pd.read_csv('NVDA.csv')

# Get the number of trading days in the data set

print(df.shape())