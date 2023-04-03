import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
import re
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

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
temp = snow_data.copy()

snow_data['country'] = pd.factorize(snow_data['country'])[0] + 1
snow_data['year'] = pd.DatetimeIndex(snow_data['date']).year
snow_data['month'] = pd.DatetimeIndex(snow_data['date']).month

X = snow_data.copy()
del X['date']

temp2 = X.copy()
del X['snow_depth']

y = snow_data[['snow_depth']]

st.title("Exploratory Data Analysis📊")

st.header("Dataset Analysis✍🏻")
st.write("The data contained in this dataset was obtained from the Climate Data Online (CDO) database maintained by the National Centers for Environmental Information (NCEI).")
st.write("The dataset comprises of average daily precipitation and air temperature data, with measurements in metric units. The original data, sourced from the CDO website, includes approximately 4.9 million data points from 1306 unique weather stations spread across Finland, Sweden and Norway.")
st.write(temp.head(5))
st.markdown("* **country**: The country of the recording.")
st.markdown("* **date**: The date of the recording.")
st.markdown("* **precipitation**: Amount of precipitation in centimeters.")
st.markdown("* **snow_depth**: Height of snow on the ground in millimeters.")
st.markdown("* **tavg**: Country average of daily mean temperatures in degrees Celsius.")
st.markdown("* **tmax**: Country average of daily maximum temperatures in degrees Celsius.")
st.markdown("* **tmin**: Country average of daily minimum temperatures measure in degrees Celsius.")
st.write("As you will soon see, the dataset for the purpose of a better model will change into the following.")
st.write(temp2.head(5))
st.write("To use date, we imputed it into an integer year, month and day. Though we later drop day due to significance as seen later.")
st.write("We also classify countries (Finland, Sweden, Norway) into integers (1, 2, 3) to allow to be used by the model.")
st.write("There was no need for any data inserting/removal for this dataset.")
st.header("Feature Importance and Significance💯")

sel_col, disp_col = st.columns(2)

rfc = joblib.load("./random_forest.joblib")

importances = rfc.feature_importances_
feature_names = X.columns

values = [['country', 0.11617713813823637],['precipitation', 0.013456622308792814], ['tavg',  0.032411218402021075], ['tmax', 0.01809914358526265], ['tmin', 0.04324784297501692], ['year', 0.08006693939830328], ['month', 0.6965410951923667], ['day', 0.0028]]

test = pd.DataFrame(values, columns=['Feature', 'Importance'])

sel_col.subheader("Random Forest Importance")
sel_col.write("Random forest feature importance is a method used in machine learning to estimate the importance of features in a dataset based on the predictive performance of a random forest model. Higher importance number the more significant.")
sel_col.write(test)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd



# Initialize variables
best_features = []
best_score = 0

# Loop through features
for feature in X.columns:
    # Add feature and train model
    features = best_features + [feature]
    X_train = X[features]
    model = LinearRegression()
    score = cross_val_score(model, X_train, y, cv=5).mean()
    
    # Check if score is better than current best score
    if score > best_score:
        best_features = features
        best_score = score
        
# Print results
disp_col.subheader("Forward Feature Importance")
disp_col.write("Forward feature selection is a feature selection method used in machine learning to select the best subset of features by iterating through feature subsets.  The features below are the ones chosen by the selection.")
disp_col.write(best_features)
st.write("Since we can see that the 'day' variable is overly insignificant, for a simpler and better performaing model, we removed 'day' from the prediction and trainig process.")

st.header("Feature Visualization📈")
st.write("Below we will be able to see feature visualizations relative to the target variable (snow_depth)")
st.write("This next plot will show feature correlation. The correlation value repressents how the target should value given the feature value. If there is a negative corrrelation, it means there is a negative linear relationship between the feature and the target variable")

corr_matrix = temp2.corr()

# Extract the correlations between the features and the target variable
corr_with_target = corr_matrix["snow_depth"].sort_values(ascending=False)

# Create a bar plot to visualize the correlations
fig, ax = plt.subplots()
sns.barplot(x=corr_with_target.values, y=corr_with_target.index)
plt.title("Feature Correlation with Target Variable")
plt.xlabel("Correlation")
plt.ylabel("Features")
plt.xticks(rotation=45)

# Display the plot in Streamlit
st.pyplot(fig)

st.write("These next smaller graphs show that same correlation but using points on a graph.")
st.write("We can see how the negative correlated graphs slightly skew to the left when a median linear line is drawn.")

# Create scatter plots of each feature with the target variable
fig, axs = plt.subplots(ncols=len(temp2.columns)-1, figsize=(20, 5))
for i, feature in enumerate(temp2.columns[:-1]):
    sns.scatterplot(x=temp2[feature], y=temp2["snow_depth"], ax=axs[i])
    axs[i].set_title(f"{feature} vs. snow_depth")
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel("snow_depth")

# Display the plots in Streamlit
st.pyplot(fig)

st.write("We can see the same correlation value but now amongst all variables.")
st.write("We can see that the temperature variables (tavg, tmin, tmax) are very similar since they all roughly record the same data.")

corr = temp2.corr()

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)

# Add title and axis labels
ax.set_title('Feature Correlation to Target Variable')
ax.set_xlabel('Features')
ax.set_ylabel('Features')

st.pyplot(fig)
