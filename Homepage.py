import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
import re
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


header = st.container()
features = st.container()
dataset = st.container()

with header:
    st.title("Welcome to :blue[MSM]!")
    st.subheader("What is :blue[MSM]❓")
    st.write("MSM stands for Mitchell's Snow Model.")
    st.subheader("What is the model goal?❄️")
    st.write("This model predicts snow depth from numerous locations is Finland, Sweden and Norway.")
    st.subheader("How can I use this python web app?🐍")
    st.write("Using the expandable sidebar, you can travel through pages to see how the model was built, exploratory data analysis and more!")
    st.write("See below for more information about the app's pages")


with features:
    st.header(":blue[MDM] Pages📄")
    st.subheader("Model🤖")
    st.markdown("* Model development")
    st.markdown("* Prediction entry")
    st.markdown("* Model evaluation")

    st.subheader("EDA📊")
    st.markdown("* Exploratory data analysis")
    st.markdown("* Feature development")
    st.markdown("* Feature importance")

    st.subheader("Resources📙")
    st.markdown("* Dataset link")
    st.markdown("* Streamlit resources")

with dataset:
    st.header("How to view other pages in Streamlit❓")
    st.write("Using the right facing arrow in the top left, you can expand the sidebar and select your desired page.")

