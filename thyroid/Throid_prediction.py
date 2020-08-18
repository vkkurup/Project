import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Lambda, Reshape,Flatten

import scikitplot as skplt

from sklearn.metrics import precision_recall_fscore_support as score


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error

st.write("""
# Thyroid prediction 
This app predicts the **thyroid status of a patient**!
""")

st.sidebar.header('User Input Parameters')




def user_input_features():
    age = st.sidebar.slider('age', 1, 120, 53)
    TSH = st.sidebar.slider('TSH', 0.0, 1000.00, 8.10)
    T3 = st.sidebar.slider('T3', 0.0, 20.0, 2.4)    
    TT4 = st.sidebar.slider('TT4', 0.0, 530.0, 132.0)
    T4U = st.sidebar.slider('T4U', 0.0, 5.0, 1.28)
    FTI = st.sidebar.slider('FTI', 0.0, 500.0, 103.0)  
    
    sex_F = st.sidebar.radio(label="sex_F", options=[0, 1])            
    sex_M = st.sidebar.radio(label="sex_M", options=[0, 1])   
    
    on_thyroxine = st.sidebar.radio(label="on thyroxine", options=[0, 1])    
    query_on_thyroxine = st.sidebar.radio(label="query on thyroxine", options=[0, 1])     
    on_antithyroid_medication = st.sidebar.radio(label="on antithyroid medication", options=[0, 1])  
    sick = st.sidebar.radio(label="sick", options=[0, 1])   
    pregnant = st.sidebar.radio(label="pregnant", options=[0, 1])    
    thyroid_surgery = st.sidebar.radio(label="thyroid surgery", options=[0, 1])     
    I131_treatment = st.sidebar.radio(label="I131 treatment", options=[0, 1])    
    query_hypothyroid = st.sidebar.radio(label="query hypothyroid", options=[0, 1])   
    query_hyperthyroid = st.sidebar.radio(label="housing", options=[0, 1])     
    lithium = st.sidebar.radio(label="lithium", options=[0, 1])     
    goitre = st.sidebar.radio(label="goitre", options=[0, 1])   
    tumor = st.sidebar.radio(label="tumor", options=[0, 1])       
    hypopituitary = st.sidebar.radio(label="hypopituitary", options=[0, 1])   
    psych = st.sidebar.radio(label="psych", options=[0, 1])    
    TSH_measured = st.sidebar.radio(label="TSH measured", options=[1, 0])   
    T3_measured = st.sidebar.radio(label="T3 measured", options=[1, 0])     
    TT4_measured = st.sidebar.radio(label="TT4 measured", options=[1, 0])       
    T4U_measured = st.sidebar.radio(label="T4U measured", options=[1, 0])   
    FTI_measured = st.sidebar.radio(label="FTI measured", options=[1, 0])
    
    data = {'age': age,
            'TSH': TSH,
            'T3': T3,
            'TT4': TT4,
            'T4U': T4U,
            'FTI': FTI,
            'sex_F': sex_F,  
            'sex_M': sex_M,              
            'on thyroxine': on_thyroxine,
            'query on thyroxine': query_on_thyroxine,
            'on antithyroid medication': on_antithyroid_medication,    
            'sick': sick,
            'pregnant': pregnant,
            'thyroid surgery': thyroid_surgery,
            'I131 treatment': I131_treatment,
            'query hypothyroid': query_hypothyroid,
            'query hyperthyroid': query_hyperthyroid,
            'lithium': lithium,
            'goitre': goitre,
            'tumor': tumor,
            'hypopituitary': hypopituitary,
            'psych': psych,
            'TSH measured': TSH_measured,
            'T3 measured': T3_measured,
            'TT4 measured': TT4_measured,
            'T4U measured': T4U_measured,
            'FTI measured': FTI_measured
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()





st.subheader('User Input parameters ')


st.write(df)




X_test = pd.read_csv("X_test.csv", sep=','  , engine='python')

y_test = pd.read_csv("y_test.csv", sep=','  , engine='python')
results = pd.DataFrame(columns=['model', 'accuracy', 'precision','recall', 'fscore'])

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Random Forest CLASSIFIER$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

st.header('Predictions')
# Random Forest Prediction
# Reads in saved regression model
loaded_RFC_clf = joblib.load(open('VK_RFC_Model.pkl', 'rb'))

RFC_prediction = loaded_RFC_clf.predict(df)
st.subheader('Random Forest classifier')
classifier = 'RandomForestClassifier'
st.write(RFC_prediction)


RFC_pred = loaded_RFC_clf.predict(X_test)


RFC_PRED_integer = pd.DataFrame(RFC_pred.round())

cm = confusion_matrix(y_test,RFC_PRED_integer)

st.write('CM')

st.write(cm)

accuracy= accuracy_score(y_test, RFC_PRED_integer)
precision,recall,fscore,support=score(y_test,RFC_PRED_integer,average='macro')


MCC= matthews_corrcoef(y_test, RFC_PRED_integer)

Aresult = [[classifier,accuracy,precision,recall,fscore,MCC]]
oneresultdf = pd.DataFrame(Aresult,columns=['model', 'accuracy', 'precision','recall', 'fscore','MCC'])

results.append(oneresultdf)
results=pd.concat([results, oneresultdf])

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Random Forest regression$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Random Forest Prediction
# Reads in saved regression model
loaded_RF_clf = joblib.load(open('VK_RF_Model.pkl', 'rb'))
classifier = 'Random Forest regression'
RF_prediction = loaded_RF_clf.predict(df)
st.subheader('Random Forest regression')
st.write(RF_prediction)


RFR_pred = loaded_RF_clf.predict(X_test)


RFR_PRED_integer = pd.DataFrame(RFR_pred.round())

cm = confusion_matrix(y_test,RFR_PRED_integer)

st.write('CM')

st.write(cm)

accuracy= accuracy_score(y_test, RFR_PRED_integer)
precision,recall,fscore,support=score(y_test,RFR_PRED_integer,average='macro')


MCC= matthews_corrcoef(y_test, RFR_PRED_integer)

Aresult = [[classifier,accuracy,precision,recall,fscore,MCC]]
oneresultdf = pd.DataFrame(Aresult,columns=['model', 'accuracy', 'precision','recall', 'fscore','MCC'])

results.append(oneresultdf)
results=pd.concat([results, oneresultdf])

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$sequential$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

model = Sequential()
# first layer must have a defined input shape
model.add(Dense(32, input_dim=27))
# afterwards, Keras does automatic shape inference
model.add(Dense(32,activation = 'relu'))

model.add(Dense(16,activation = 'relu'))

model.add(Dense(8,activation = 'relu'))

model.add(Dense(4,activation = 'relu'))


model.add(Dense(2,activation = 'softmax'))

model.compile(loss="sparse_categorical_crossentropy", 
              optimizer="adam",
              metrics=["accuracy"])
              
model.load_weights('model_sequential.h5')
classifier = 'SEQUENCER'


sequential_prediction = model.predict(df)
st.subheader('sequential_prediction')

st.write(sequential_prediction[:,1])




sty_pred = model.predict_proba(X_test)

sty_float = sty_pred[:,1]
sty_integer = pd.DataFrame(sty_float.round())


cm = confusion_matrix(y_test,sty_integer)

st.write('CM')

st.write(cm)


accuracy= accuracy_score(y_test, sty_integer)
precision,recall,fscore,support=score(y_test,sty_integer,average='macro')


MCC= matthews_corrcoef(y_test, sty_integer)

Aresult = [[classifier,accuracy,precision,recall,fscore,MCC]]
oneresultdf = pd.DataFrame(Aresult,columns=['model', 'accuracy', 'precision','recall', 'fscore','MCC'])

results.append(oneresultdf)
results=pd.concat([results, oneresultdf])



st.write('COMPARISON')

st.write(results)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$sequential$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#y_prdicted_1_values = RFR_PRED_integer.loc[RFR_PRED_integer==1]


y_prdicted_1_values = X_test[(sty_integer>0).values]
y_prdicted_1_values.columns = [df.columns]

st.write('there are few rows having predicted 1....... they are the ones belowy_prdicted_1_values')

st.write(y_prdicted_1_values)


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$DATASET USED FOR PREDICTION$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Y_test = pd.read_csv("Y_test.csv", sep=','  , engine='python')


st.write('X_TEST')

st.write(X_test)


st.write('y_TEST')

st.write(y_test)



st.write('RANDOM FOREST  CLASSIFICATION PREDITED VALUES')

st.write(RFC_PRED_integer)



st.write('RANDOM FOREST  REGRESSION PREDITED VALUES')

st.write(RFR_PRED_integer)



st.write('SEQUENTIAL  PREDITED VALUES')

st.write(sty_integer)

