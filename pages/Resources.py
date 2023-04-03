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

st.header("Resources📙")
st.subheader("Dataset Link🔗")
st.write("Click this [link](https://www.kaggle.com/datasets/adamwurdits/finland-norway-and-sweden-weather-data-20152019) to see dataset on Kaggle.")

st.subheader("Streamlit Turorial🖥️")
st.write("Click this [link](https://www.youtube.com/watch?v=-IM3531b1XU&ab_channel=M%C4%B1sraTurp) to for a detailed streamlit tutorial series.")

st.subheader("Streamlit Homepage🏠")
st.write("Click this [link](https://streamlit.io/) to for a detailed streamlit tutorial series.")

