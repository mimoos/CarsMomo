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
## Car_mileage = int(Car_mileage)

## To get dummies for Make and Model

## columns = pd.get_dummies(data=Car_data, columns=['Make', 'Model'])        too large data >200mb
column_Make = pd.get_dummies(data=Car_data, columns=['Make'])


label_encoder = preprocessing.LabelEncoder()
column_Model = label_encoder.fit_transform(Car_data['Model'])
column_Model = pd.DataFrame(column_Model)
column_Model

## Car_model = label_encoder.transform([Car_model])

## Car_data_new = Car_data.drop(['Make', 'Model'], axis=1)
Car_data_new = pd.concat([Car_data_new.reset_index(drop=True), column_Make.reset_index(drop=True), column_Model.reset_index(drop=True)], axis=1)

Car_data_new

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
