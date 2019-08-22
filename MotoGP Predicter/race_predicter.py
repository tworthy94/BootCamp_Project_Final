# Import dependencies
import os
import time
import pandas as pd
import numpy as np
import csv
import operator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder

 # Create Label Encoders
lbl = LabelEncoder()
lbl2 = LabelEncoder()
lbl3 = LabelEncoder()
lbl4 = LabelEncoder()

# Create model
model = LinearRegression()

# Model training function
def trainModel(file):
    # Training file
    df_training = pd.read_csv(file, index_col=0)

    # Encoding string based columns
    df_training["Rider_Name"] = lbl.fit_transform(df_training["Rider_Name"])
    df_training["Bike"] = lbl2.fit_transform(df_training["Bike"])
    df_training["Track_Condition"] = lbl3.fit_transform(df_training["Track_Condition"])
    df_training["Track"] = lbl4.fit_transform(df_training["Track"])
    df_training = df_training[(df_training["Finish_Time_ms"]>=(20*60000))]
    df_training = df_training[(df_training["Category"] == "MotoGP")]
    df_training = df_training.drop(columns=['Category'])


    x = df_training.drop("Finish_Time_ms", axis = 1)
    y = df_training["Finish_Time_ms"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    polynomial_features= PolynomialFeatures(degree=3)
    x_poly = polynomial_features.fit_transform(x_train)
    # model = LinearRegression()
    model.fit(x_poly, y_train)
    y_poly_pred = model.predict(x_poly)
    rmse = np.sqrt(mean_squared_error(y_train,y_poly_pred))
    r2 = r2_score(y_train,y_poly_pred)
    print(f"Training Set RMSE Score: {rmse}")
    print(f"Training Set R2 Score:  {r2}")

    # Testing
    x_poly_test = polynomial_features.fit_transform(x_test)
    y_poly_pred = model.predict(x_poly_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_poly_pred))
    r2 = r2_score(y_test,y_poly_pred)
    print(f"Test Set RMSE Score: {rmse}")
    print(f"Test Set R2 Score:  {r2}")

    # # Create predictions dataframe
    # df_predictions = pd.DataFrame({"Actual Finish Time in Minutes": y_test, "Predicted Finish Time in Minutes": y_poly_pred})

    # # Convert miliseconds to minutes
    # df_predictions = df_predictions / 60000
    # df_predictions["Predicted Finish Time in Minutes"] = round(df_predictions["Predicted Finish Time in Minutes"],2)
    # df_predictions["Actual Finish Time in Minutes"] = round(df_predictions["Actual Finish Time in Minutes"],2)
    
    # df_predictions["Rank"] = df_predictions["Predicted Finish Time in Minutes"].rank()
    # df_predictions.round({"Rank": 0})
    # df_predictions=df_predictions.sort_values(by=["Rank"])
    # df_training["Rider_Name"] = lbl.inverse_transform(df_training["Rider_Name"])
    # df_predictions["Rider_Name"] = df_training["Rider_Name"].str.title()

    # # Print predictions
    # print(df_predictions.head(10))
      
# Model predicting function
def predictModel(file):
    # Race file
    df_race = pd.read_csv(file, index_col=0)

    df_race["Rider_Name"] = lbl.transform(df_race["Rider_Name"])
    df_race["Bike"] = lbl2.transform(df_race["Bike"])
    df_race["Track_Condition"] = lbl3.transform(df_race["Track_Condition"])
    df_race["Track"] = lbl4.transform(df_race["Track"])
    df_race = df_race[(df_race["Category"] == "MotoGP")]
    df_race = df_race.drop(columns=['Category'])


    x_test = df_race.drop("Finish_Time_ms", axis = 1)
    y_test = df_race["Finish_Time_ms"]
    polynomial_features= PolynomialFeatures(degree=3)
    x_poly_test = polynomial_features.fit_transform(x_test)
    y_poly_pred = model.predict(x_poly_test)

    df_predictions2 = pd.DataFrame({"Actual Finish Time in Minutes": y_test, "Predicted Finish Time in Minutes": y_poly_pred})

    # Convert miliseconds to minutes
    df_predictions2 = df_predictions2 / 60000
    df_predictions2["Predicted Finish Time in Minutes"] = round(df_predictions2["Predicted Finish Time in Minutes"],2)
    df_predictions2["Actual Finish Time in Minutes"] = round(df_predictions2["Actual Finish Time in Minutes"],2)

    df_predictions2['Rank'] = df_predictions2['Predicted Finish Time in Minutes'].rank()
    df_predictions2.round({'Rank': 0})
    df_predictions2=df_predictions2.sort_values(by=['Rank'])
    df_race["Rider_Name"] = lbl.inverse_transform(df_race["Rider_Name"])
    df_predictions2['Rider_Name'] = df_race['Rider_Name'].str.title()

    print(f"\n {df_predictions2}")