This is a project I implemented in Python for the prediction of housing price.

1. Load the data from cal_housing with header.csv
2. Process preprocessing
3. Fin correlation between the variables.
4. split the data train vs test 70-30
5. From here it takes multiple branches.
A. do Normalization B. do standardization and C.  use without both  
Use scaler information from the respective files is regularization is used.
6.  Another deviation is to Do Linear regression, Ridge Regression, Lasso Regression, Elastic net regression, Random Forest Regression, SVR and ADA Boost Regressions. Similar ones are grouped in one notebook.
7. Save the pickle file and the scaler files for the client to use for the predictions from user inputs.
8. The client is made using streamlit and the requirement file with the details the import files.
How to make it work::
Copy following files xx.pkl, scaler file xx.save, homepredictor.py.
Eg: VK_ELASTIC_Model_mm.pkl elastic = elasticnet, mm = min-max scaler CV = cross validation used etc.
Capstone California House-LR-RR-LASSO-NET - Zscore standardization.ipynb
LR – Linear regression, RR = Ridge, Lasso = lasso regression, Zscre is the standardization etc.

Import the following packages (install if they are not installed)
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
using the command prompt(with conda or any other which has the env as given in the requirements.txt.) run the following command
streamlit run homepredictor.py

Which will open the browser window and open the client for user to input the value using slide bar for various features.

The sample out is given in the jpg file
sample output.JPG

Any questions Please ask vkkurup@gmail.com
