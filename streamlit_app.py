import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

Car_data = pd.read_csv('true_car_listings_fix.csv')

## To ask the user the Make, Model, Year and Mileage

List_make = Car_data['Make'].unique().tolist()
List_make.sort()

List_model = Car_data['Model'].unique().tolist()
List_model.sort()

List_year = Car_data['Year'].unique().tolist()
List_year.sort()

Car_make = st.selectbox('Car Make:', options=List_make, index=None)
Car_model = st.selectbox('Car Model:', options=List_model, index=None)
Car_year = st.selectbox('Car Year:', options=List_year, index=None)
Car_mileage = st.text_input('Car Mileage:', value=None)

st.button('Predict the price')

## To get dummies for Make and encoding for Model

## columns = pd.get_dummies(data=Car_data, columns=['Make', 'Model'])        too large data >200mb
column_Make = pd.get_dummies(data=Car_data, columns=['Make'])
column_Make = column_Make.drop(['Price', 'Year', 'Model', 'Mileage'], axis=1)

## label_encoder = preprocessing.LabelEncoder()
## column_Model = label_encoder.fit_transform(Car_data['Model'])
One_Hot_Encoder = preprocessing.OneHotEncoder()
column_Model = Car_data['Model']
column_Model = One_Hot_Encoder.fit_transform(column_Model)
## column_Model = pd.DataFrame(column_Model, columns = ['Model'])

Column_Model

Car_data_new = Car_data.drop(['Make', 'Model'], axis=1)
Car_data_new = pd.concat([Car_data_new.reset_index(drop=True), column_Make.reset_index(drop=True), column_Model.reset_index(drop=True)], axis=1)

Car_data_new

## MACHINELEARNING

y = Car_data_new['Price']
X = Car_data_new.drop(['Price'], axis=1)

LinearModel = LinearRegression()
LinearModel.fit(X, y)

X_Features = X.columns

b = LinearModel.intercept_
w = LinearModel.coef_

w

def Click_Predict (Car_make, Car_model, Car_year, Car_mileage):
    data = [None] * 61

    Column_Names = []
    for row in X_Features:
        Column_Names.append(row)

    ## Car_year = int(Car_year)
    data[0]= Car_year

    Car_mileage = int(Car_mileage)
    data[1] = Car_mileage

    Car_make = 'Make_' + Car_make
    Column_Names = np.array(Column_Names)
    # for index, i in np.ndenumerate(Column_Names):
    for i in range(Column_Names.shape[0]):  # or range(len(theta))
    # for i in enumerate(Column_Names):
        if Column_Names[i] == Car_make:
            data[i] = 1
        
    Car_model = label_encoder.transform([Car_model])
    data[60] = int(Car_model)
    
    ## df = pd.DataFrame(data)
    ## price_predicted
    return data

See = Click_Predict(Car_make, Car_model, Car_year, Car_mileage)


X.shape[0]
X_Features[60]
See


"""
# Welcome to Streamlit CAN I EDIT THIS?!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

indices = np.linspace(0, 1, num_points)
theta = 2 * np.pi * num_turns * indices
radius = indices

x = radius * np.cos(theta)
y = radius * np.sin(theta)

df = pd.DataFrame({
    "x": x,
    "y": y,
    "idx": indices,
    "rand": np.random.randn(num_points),
})

st.altair_chart(alt.Chart(df, height=700, width=700)
    .mark_point(filled=True)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        color=alt.Color("idx", legend=None, scale=alt.Scale()),
        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
    ))
