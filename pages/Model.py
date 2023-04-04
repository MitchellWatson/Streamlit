import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="MSM App",
    page_icon="❄️"
)

snow_data = pd.read_csv("nordics_weather.csv")

snow_data['country'] = pd.factorize(snow_data['country'])[0] + 1
snow_data['year'] = pd.DatetimeIndex(snow_data['date']).year
snow_data['month'] = pd.DatetimeIndex(snow_data['date']).month

X = snow_data.copy()
del X['date']
del X['snow_depth']

y = snow_data[['snow_depth']]

st.title("Model page")

st.header("Model Development🛠️")
st.write("The model being used to predict snow depth is a RandomForestRegressor.")
st.write("RandomForestRegressor fits a number of classifying decision tees to various sub-samples.")
st.write("Many continuous data prediction model were tested such sa LinearRegression, XGBoost and more.")
st.write("RandomForestRegression proved to be the best using RMSE as performance metric.")
st.write("This RandomForestRegression contains:")
st.markdown("* 1000 estimators")
st.markdown("* Default tree depth")
st.markdown("* Default max features")


st.header("Model Evaluation📝")
st.write("Using the nordics_weather.csv, we can evaluate the model's performance")
st.write("The dataset uses recordings and data from numerous different locations among different nordic countries.")
st.write("This makes it so that a recording in located Finland could be geographically closer to a recording in Sweden compared to a different recording from Finland.")
st.write("Therefore, this causes some unfortunate performance costs to the model as seen below.")

st.subheader("R2 Score")
st.write(0.9552228488958796)

st.subheader("RMSE")
st.write(37.01933497965017)


modelTraining = st.container()    


with modelTraining:
    st.header("Model predictions🧐")
    st.write("Input your parameters to see how much snow depth the model will predict!")

    sel_col, disp_col = st.columns(2)

    country_choice = sel_col.selectbox("Country", options=["Finland", "Norway", "Sweden"], index=0 )

    month = sel_col.slider('Month', min_value=1, max_value=12, value=1, step=1)
    
    year = sel_col.slider('Year', min_value=2015, max_value=2019, value=2015, step=1)

    precipitation = sel_col.text_input('Precipitation', '')

    average = sel_col.text_input('Average Daily Temperature (°C)', '')
    
    minimum = sel_col.text_input('Minimum Daily Temperature (°C)', '')

    maximum = sel_col.text_input('Maximum Daily Temperature (°C)', '')

    submit = sel_col.button('Predict')

    if (submit):
        if (average and minimum and precipitation and maximum and month and year and country_choice): 
            try:
                if (country_choice == "Finland"):
                    country_choice = 1
                elif (country_choice == "Norway"):
                    country_choice = 2
                else:
                    country_choice = 3       

                values = [country_choice, precipitation, average, maximum, minimum, year, month]

                test = pd.DataFrame([values], columns=['country', 'precipitation', 'tavg', 'tmax', 'tmin', 'year', 'month'])

                model = joblib.load("./random_forest.joblib")

                predictions = model.predict(test)

                disp_col.subheader("Prediction (mm)")
                disp_col.write(predictions)

                sel_col.subheader(":green[Successfully displaying prediction in the above right!]")    

            except:
                sel_col.subheader(":red[write inputs must be integer or float values]")    

        else :
            sel_col.subheader(":red[Please input all parameters above]")    
        

