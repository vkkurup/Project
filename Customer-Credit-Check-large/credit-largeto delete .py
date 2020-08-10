import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

st.write("""
# California home Prediction App
This app predicts the **Home Price **!
""")

st.sidebar.header('User Input Parameters')


'age', 'balance', 'duration', 'defaultcat', 'previous', 'poutcome', 'y',
       'jobcat', 'educationcat', 'defaultcat', 'housingcat
def user_input_features():
    age = st.sidebar.slider('age', -124.350000, -114.310000, -119.569704)
    balance = st.sidebar.slider('balance', 35.631861, 41.950000, 35.631861)
    duration = st.sidebar.slider('duration', 1, 52, 28)
    previous = st.sidebar.slider('previous', 2, 39320, 2635)
    poutcome = st.sidebar.slider('poutcome', 1, 6445, 537)
    defaultcat = st.sidebar.slider('defaultcat', 3, 35682, 1425)
    jobcat = st.sidebar.slider('jobcat', 1, 6082, 499)
    educationcat = st.sidebar.slider('educationcat',0.499900, 15.000100, 3.870671)    
    
    defaultcat = st.sidebar.slider('defaultcat', 3, 35682, 1425)
    housingcat = st.sidebar.slider('housingcat', 1, 6082, 499)
    data = {'age': age,
            'balance': balance,
            'duration': duration,
            'previous': previous,
            'poutcome': poutcome,
            'defaultcat': defaultcat,
            'jobcat': jobcat,
            'educationcat': educationcat,
            'defaultcat': defaultcat,
            'housingcat': housingcat
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('User Input parameters nonscaled')
st.write(df_nonscaled)






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$mno scale belowbelow$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

st.header('WITHOUT NORMALIZATION OR STANDARDIZATION')
# Random Forest Prediction
# Reads in saved regression model
loaded_LinR_clf = joblib.load(open('VK_linR_Model_sm.pkl', 'rb'))

LinR_prediction = loaded_LinR_clf.predict(df_nonscaled)
st.subheader('LinearRegression')
st.write(LinR_prediction)


# Linear regression
# Reads in saved regression model
loaded_XGB_clf = joblib.load(open('VK_xgb_Model_sm.pkl', 'rb'))

xgb_prediction = loaded_XGB_clf.predict(df_nonscaled)

st.subheader('XGBClassifier')
st.write(xgb_prediction)


# RIDGE regression
# Reads in saved regression model
loaded_LogR_clf = joblib.load(open('VK_logR_Model_sm.pkl', 'rb'))

LogR_prediction = loaded_LogR_clf.predict(df_nonscaled)
st.subheader('LogisticRegression')
st.write(LogR_prediction)



# LASSO regression
# Reads in saved regression model
loaded_DTR_clf = joblib.load(open('VK_DTR_Model_sm.pkl', 'rb'))

DTR_prediction = loaded_DTR_clf.predict(df_nonscaled)
st.subheader('DecisionTreeRegressor')
st.write(DTR_prediction)





# Linear regression
# Reads in saved regression model
loaded_DTC_clf = joblib.load(open('VK_DTC_Model_sm.pkl', 'rb'))

dtc_prediction = loaded_DTC_clf.predict(df_nonscaled)

st.subheader('DecisionTreeClassifier')
st.write(dtc_prediction)



# Linear regression
# Reads in saved regression model
loaded_RF_clf = joblib.load(open('VK_RF_Model_sm.pkl', 'rb'))

rf_prediction = loaded_RF_clf.predict(df_nonscaled)

st.subheader('RandomForestRegressor')
st.write(rf_prediction)


# Linear regression
# Reads in saved regression model
loaded_SVM_clf = joblib.load(open('VK_svm_Model_sm.pkl', 'rb'))

svm_prediction = loaded_SVM_clf.predict(df_nonscaled)

st.subheader('SVM')
st.write(svm_prediction)


# Linear regression
# Reads in saved regression model
loaded_ADB_clf = joblib.load(open('VK_ADB_Model_sm.pkl', 'rb'))

adb_prediction = loaded_ADB_clf.predict(df_nonscaled)

st.subheader('XGBClassifier')
st.write(adb_prediction)









