# Importing libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Loading the dataset
fp= open(r"C:\Users\nsnar\Downloads\car_data.csv", "r")
df = pd.read_csv(fp)

# Preprocessing the data
df.drop_duplicates(inplace=True)
df.drop(['Car_Name'], axis=1, inplace=True)
df['Age'] = 2023 - pd.to_datetime(df['Year']).dt.year
df.drop(['Year'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Splitting the dataset
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Root Mean Squared Error: ", rmse)
print("R2 Score: ", r2)

# Predicting the selling price of a used car
test_data = np.array([50000, 5, 3, 1, 0, 0, 0, 1, 1, 0]).reshape(1,-1)
selling_price = model.predict(test_data)
print("Predicted Selling Price: ", selling_price)

