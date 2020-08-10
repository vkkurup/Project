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
    age = st.sidebar.slider('age', 18, 95, 20)
    balance = st.sidebar.slider('balance', -8019.00, 102127.00, 13620.00)
    duration = st.sidebar.slider('duration', 0, 4918, 0)
    
    pdays = st.sidebar.slider('pdays', -1, 530, -1)
    previous = st.sidebar.slider('previous', 0.0, 275.0, 0.0)
    poutcome = st.sidebar.slider('poutcome', 0.0, 3.0, 0.0)
    
    
    job = st.sidebar.radio(label='job',options=["unemployed", "student", "unknown", "retired","housemaid","blue-collar", 
        "technician","services", "admin.","self-employed","management","entrepreneur" ])
     
    education = st.sidebar.radio(label="education", options=["primary", "unknown", "secondary", "tertiary"]) 
    
    default = st.sidebar.radio(label="default", options=["no", "yes"])
    
    housing = st.sidebar.radio(label="housing", options=["no", "yes"])
    
    data = {'age': age,
            'balance': balance,
            'duration': duration,
            'previous': previous,
            'poutcome': poutcome,
            'default': default,
            'job': job,
            'education': education,
            'pdays': pdays,
            'housing': housing
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()




df["education"] = df["education"].astype('category')
df["default"] = df["default"].astype('category')
df["housing"] = df["housing"].astype('category')

df["job"] = df["job"].astype('category')

df['jobcat'] = df['job'].apply(lambda x: ['unemployed', 'student', 'unknown', 'retired','housemaid','blue-collar', 'technician','services', 'admin.','self-employed','management','entrepreneur' ].index(x))
df['educationcat'] = df['education'].apply(lambda x: ['primary', 'unknown', 'secondary', 'tertiary'].index(x))
df['defaultcat'] = df['default'].apply(lambda x: ['no', 'yes'].index(x))
df['housingcat'] = df['housing'].apply(lambda x: ['no', 'yes'].index(x))






del df['job']
del df['education']
del df['default']
del df['housing']



st.subheader('User Input parameters nonscaled')




df['jobcat']= df['jobcat'].cat.codes
df['educationcat']= df['educationcat'].cat.codes
df['housingcat']= df['housingcat'].cat.codes
df['defaultcat']= df['defaultcat'].cat.codes

st.write(df)
predictors = list(list(df.columns))
df = df[predictors].values

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$mno scale belowbelow$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

st.header('WITHOUT NORMALIZATION OR STANDARDIZATION')
# Random Forest Prediction
# Reads in saved regression model
loaded_LinR_clf = joblib.load(open('VK_linR_Model.pkl', 'rb'))

LinR_prediction = loaded_LinR_clf.predict(df)
st.subheader('LinearRegression')
st.write(LinR_prediction)


# Linear regression
# Reads in saved regression model
loaded_XGB_clf = joblib.load(open('VK_xgb_Model.pkl', 'rb'))

xgb_prediction = loaded_XGB_clf.predict(df)

st.subheader('XGBClassifier')
st.write(xgb_prediction)


# RIDGE regression
# Reads in saved regression model
loaded_LogR_clf = joblib.load(open('VK_logR_Model.pkl', 'rb'))

LogR_prediction = loaded_LogR_clf.predict(df)
st.subheader('LogisticRegression')
st.write(LogR_prediction)



# LASSO regression
# Reads in saved regression model
loaded_DTR_clf = joblib.load(open('VK_DTR_Model.pkl', 'rb'))

DTR_prediction = loaded_DTR_clf.predict(df)
st.subheader('DecisionTreeRegressor')
st.write(DTR_prediction)





# Linear regression
# Reads in saved regression model
loaded_DTC_clf = joblib.load(open('VK_DTC_Model.pkl', 'rb'))

dtc_prediction = loaded_DTC_clf.predict(df)

st.subheader('DecisionTreeClassifier')
st.write(dtc_prediction)



# Linear regression
# Reads in saved regression model
loaded_RF_clf = joblib.load(open('VK_RF_Model.pkl', 'rb'))

rf_prediction = loaded_RF_clf.predict(df)

st.subheader('RandomForestRegressor')
st.write(rf_prediction)


# # Linear regression
# # Reads in saved regression model
# loaded_SVM_clf = joblib.load(open('VK_svm_Model.pkl', 'rb'))

# svm_prediction = loaded_SVM_clf.predict(df)

# st.subheader('SVM')
# st.write(svm_prediction)


# Linear regression
# Reads in saved regression model
loaded_ADB_clf = joblib.load(open('VK_ADB_Model.pkl', 'rb'))

adb_prediction = loaded_ADB_clf.predict(df)

st.subheader('XGBClassifier')
st.write(adb_prediction)









