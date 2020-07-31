# READ ME

## Table of Contents:
(Links to different parts of the README, all throughout the github project)
[Link](Introduction:)
[Link](Included with the project:)
[Link](Getting Started:)
[Link](Usage:)
[Link](Contact:)
[Link](Acknowledgements:)
## Introduction:
"This project is for housing price prediction [application, script or something better] in the "California" area. " ADD MORE INFORMATION FOR MOTIVATION AND FUTURE APPLICATIONS on what we can do with this project
(Picture or pictures with great result of your project)
### Included with the project:
(Softwares, libraries,languages/ add-ons you used and why you used it. links that they can read up on that)

## Getting Started:
have nice segue to list all the needed things for this project. mainly for people who have experience or inexperience with your aspects of your project. a lay person should be able to go down and see if they follow what you have in your project
### Pre-requisistes:
keep up with your project or list all the install versions you need
you should also have code snippets to let them run your project
any setup.sh part should be notified here.
### Installation:
```
 Import the following packages (install if they are not installed)
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
```
> using the command prompt(with conda or any other which has the env as given in the requirements.txt.) run the following command
> streamlit run homepredictor.py

## Usage:
link to the file every time just so they check it out


### Input File: 
Describe inputs and outputs (and particular things to keep in mind like row weights n), mention how flexible your code is. 
```
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

```
![Image](https://raw.githubusercontent.com/vkkurup/Project/master/Cal-state-house-price-predictor/sample%20output.JPG)

## Contact:
Any questions Please contact Vijayakumar Kurup vkkurup@gmail.com

## Acknowledgements:

