# import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


# load the data
data = pd.read_csv("https://raw.githubusercontent.com/nfonsang/ML_App/main/house_price_data.csv")
## drop the "No" column
data = data.drop("No", axis="columns")
print(data.head())

#Extract the input and output data
x_cols = ["house_age", "distance_to_nearest_MRT_station",
           "convenience_stores"]

X = data[x_cols]
y = data["house_price_of_unit_area"]

#Split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.20,
                                                   random_state=42)

# Construct the Model
lin_reg = LinearRegression()
lin_reg = lin_reg.fit(X_train, y_train)

# Model Evaluation
r_squared_train = lin_reg.score(X_train, y_train)
r_squared_test = lin_reg.score(X_test, y_test)

print("R-squared on training set: ", r_squared_train)
print("R-squared on test set: ",  r_squared_test)


# Save the model and name it lin_reg.pkl
joblib.dump(lin_reg, "lin_reg.pkl")


