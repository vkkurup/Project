import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

st.write("""
# California home Prediction App
This app predicts the **Home Price **!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    longitude = st.sidebar.slider('longitude', -124.350000, -114.310000, -119.569704)
    latitude = st.sidebar.slider('latitude', 35.631861, 41.950000, 35.631861)
    housingMedianAge = st.sidebar.slider('housingMedianAge', 1, 52, 28)
    totalRooms = st.sidebar.slider('totalRooms', 2, 39320, 2635)
    totalBedrooms = st.sidebar.slider('totalBedrooms', 1, 6445, 537)
    population = st.sidebar.slider('population', 3, 35682, 1425)
    households = st.sidebar.slider('households', 1, 6082, 499)
    medianIncome = st.sidebar.slider('medianIncome',0.499900, 15.000100, 3.870671)
    data = {'longitude': longitude,
            'latitude': latitude,
            'housingMedianAge': housingMedianAge,
            'totalRooms': totalRooms,
            'totalBedrooms': totalBedrooms,
            'population': population,
            'households': households,
            'medianIncome': medianIncome
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
df_nonscaled = df.copy()
df_minmax = df.copy()

df_Zscore = df.copy()
df_XGB = df.copy()
st.subheader('User Input parameters nonscaled')
st.write(df_nonscaled)
# Reads in saved scaler
scaler = joblib.load(open('scaler_nm.save', 'rb'))
df_minmax=scaler.transform(df_minmax)
df_minmax = pd.DataFrame(df_minmax)
df_minmax.columns =['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
       'totalBedrooms', 'population', 'households', 'medianIncome']

st.subheader('User Input parameters after min max scaling ')
st.write(df_minmax)




# Reads in saved scaler
scaler = joblib.load(open('scaler_standardscaler.save', 'rb'))
df_Zscore=scaler.transform(df_Zscore)
df_Zscore = pd.DataFrame(df_Zscore)
df_Zscore.columns =['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
       'totalBedrooms', 'population', 'households', 'medianIncome']

st.subheader('User Input parameters after Z score scaling')
st.write(df_Zscore)




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$mno scale belowbelow$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

st.header('WITHOUT NORMALIZATION OR STANDARDIZATION')
# Random Forest Prediction
# Reads in saved regression model
loaded_RF_clf = joblib.load(open('VK_RF_Model all features.pkl', 'rb'))

rf_prediction = loaded_RF_clf.predict(df_nonscaled)
st.subheader('Random Forest Prediction df_nonscaled')
st.write(rf_prediction)


# Linear regression
# Reads in saved regression model
loaded_LR_clf = joblib.load(open('VK_lr_model.pkl', 'rb'))

lr_prediction = loaded_LR_clf.predict(df_nonscaled)

st.subheader('Linear Regression Prediction df_nonscaled')
st.write(lr_prediction)


# RIDGE regression
# Reads in saved regression model
loaded_RR_clf = joblib.load(open('VK_RR_Model.pkl', 'rb'))

RR_prediction = loaded_RR_clf.predict(df_nonscaled)
st.subheader('RIDGE Regression Prediction df_nonscaled')
st.write(RR_prediction)



# LASSO regression
# Reads in saved regression model
loaded_LASSO_clf = joblib.load(open('VK_LASSO_Model.pkl', 'rb'))

LASSO_prediction = loaded_LASSO_clf.predict(df_nonscaled)
st.subheader('LASSO Regression Prediction df_nonscaled')
st.write(LASSO_prediction)
st.write(LASSO_prediction)



# XG Boost Prediction

# loaded_XGB_clf = joblib.load(open('VK_xgboost_model.pkl', 'rb'))
# del df_XGB['households']
# del df_XGB['totalRooms']
# del df_XGB['totalBedrooms']
# xgb_prediction = loaded_XGB_clf.predict(df_XGB)
# st.subheader('XGBoost no households, totalRooms and totalBedrooms & no scaling ')
# st.write(xgb_prediction)
#st.write(#prediction#)



st.header('WITH NORMALIZATION min max')
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$min max scaler below$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Linear regression
# Reads in saved regression model
loaded_LR_clf_mm = joblib.load(open('VK_lr_model_mm.pkl', 'rb'))

lr_prediction_mm = loaded_LR_clf_mm.predict(df_minmax)

st.subheader('Linear Regression Prediction df_minmax')
st.write(lr_prediction_mm)


# RIDGE regression
# Reads in saved regression model
loaded_RR_clf_mm = joblib.load(open('VK_RR_Model_mm.pkl', 'rb'))


RR_prediction_mm = loaded_RR_clf_mm.predict(df_minmax)
st.subheader('RIDGE Regression Prediction df_minmax')
st.write(RR_prediction_mm)



# LASSO regression
# Reads in saved regression model
loaded_LASSO_clf_mm = joblib.load(open('VK_LASSO_Model_mm.pkl', 'rb'))

LASSO_prediction_mm = loaded_LASSO_clf_mm.predict(df_minmax)
st.subheader('LASSO Regression Prediction df_minmax')
st.write(LASSO_prediction_mm)


# ELASTIC regression
# Reads in saved regression model
loaded_ELASTIC_clf_mm = joblib.load(open('VK_ELASTIC_Model_mm.pkl', 'rb'))

ELASTIC_prediction_mm = loaded_ELASTIC_clf_mm.predict(df_minmax)
st.subheader('ELASTIC Regression Prediction df_minmax')
st.write(ELASTIC_prediction_mm)


st.header('WITH StandardScaler STANDARDIZATION')

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Zscore standard scaler below$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Linear regression
# Reads in saved regression model
loaded_LR_clf_standardscaler = joblib.load(open('VK_lr_model_standardscaler.pkl', 'rb'))

lr_prediction_standardscaler = loaded_LR_clf_standardscaler.predict(df_Zscore)

st.subheader('Linear Regression Prediction df_Zscore')
st.write(lr_prediction_standardscaler)


# RIDGE regression
# Reads in saved regression model
loaded_RR_clf_standardscaler = joblib.load(open('VK_RR_Model_standardscaler.pkl', 'rb'))


RR_prediction_standardscaler = loaded_RR_clf_standardscaler.predict(df_Zscore)
st.subheader('RIDGE Regression Prediction df_Zscore')
st.write(RR_prediction_standardscaler)



# LASSO regression
# Reads in saved regression model
loaded_LASSO_clf_standardscaler = joblib.load(open('VK_LASSO_Model_standardscaler.pkl', 'rb'))

LASSO_prediction_standardscaler = loaded_LASSO_clf_standardscaler.predict(df_Zscore)
st.subheader('LASSO Regression Prediction df_Zscore')
st.write(LASSO_prediction_standardscaler)


# ELASTIC regression
# Reads in saved regression model
loaded_ELASTIC_clf_standardscaler = joblib.load(open('VK_ELASTIC_Model_standardscaler.pkl', 'rb'))

ELASTIC_prediction_standardscaler = loaded_ELASTIC_clf_standardscaler.predict(df_Zscore)
st.subheader('ELASTIC Regression Prediction df_Zscore')
st.write(ELASTIC_prediction_standardscaler)








#
# st.subheader('Prediction')
#
# st.pyplot(loaded_clf.predictions)
# sns.distplot(loaded_clf.predictions-y_test)
