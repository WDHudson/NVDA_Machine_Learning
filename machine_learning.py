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

# Visualize the close price data
plt.figure(figsize=(16,8))
plt.title('NVDA Closing Price')
plt.xlabel('Days')
plt.ylabel('Adjusted Closing Price')
plt.plot(df['Adj Close'])
plt.savefig('NVDA Adj Close.png')

df = df[['Adj Close']]

# Create a variable to predeict x days out into the future
future_days = 25
# Create a new column (target) shifted x days out
df['Prediction'] =  df[['Adj Close']].shift(-future_days)

# Create the feature dataset (X) and convert it to a numpy array and remove the last 'x' rows / days
x = np.array(df.drop(['Prediction'], 1))[:-future_days]

# Create the taret data (Y) and convert it to a numpy array and get all of the target values except the last X rows
y = np.array(df['Prediction'])[:-future_days]

# Split the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# Create the models
# Create the decision tree regressive model
tree = DecisionTreeRegressor().fit(x_train, y_train)
# Create the linear regression model
lr = LinearRegression().fit(x_train, y_train)

# Get the last 'x' rows of the future dataset
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

# Show the model tree prediction
tree_prediction = tree.predict(x_future)

# Show the model linear regression prediction
lr_prediction = lr.predict(x_future)

# Visualize the Tree data
predictions = tree_prediction
valid = df[x.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('NVDA Closing Price tree prediction')
plt.xlabel('Days')
plt.ylabel('Adj Close')
plt.plot(df['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.legend(['Original Data', 'Valid Data', 'Predicted Data'])
plt.savefig('NVDA Predicted Prices Tree Prediction')

# Visualize the Linear Regression data
predictions = lr_prediction
valid = df[x.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('NVDA Closing Price tree prediction')
plt.xlabel('Days')
plt.ylabel('Adj Close')
plt.plot(df['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.legend(['Original Data', 'Valid Data', 'Predicted Data'])
plt.savefig('NVDA Predicted Prices Regression Prediction')