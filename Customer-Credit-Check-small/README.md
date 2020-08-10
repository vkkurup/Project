Introduction:
"This Capstone project is created for predicting the credit eligibility for the customers. The dataset used for this project is a comma separated file called “Bank.csv with”. Several ML algorithms were used in doing this.
Included with the project:
This project includes the input dataset, the Jupyter notebook, the necessary python packages, and a client implementation.
Getting Started:
The data set has the following features
'age', 'balance', 'duration', 'pdays', 'previous', 'poutcome', 'y',
'jobcat', 'educationcat', 'defaultcat', 'housingcat'.
.
Pre-requisites:
To run this project and evaluating various algorithms, some basic computer knowledge along with knowledge of running a python project, and knowledge of prediction logics and HTML client application.
Installation:
Install the following packages
Python 3.7
streamlit
pandas
joblib
sklearn
or use the requirement.txt for the installation if IDE like PyCharm is used.
Usage:
From any command prompt like conda with environment where all the above packages given in the installation section, are installed and then run the following command
“streamlit run credit-small.py”
Input File:
There are two aspects of this project.
1.	Create the prediction model.
To do this Jupyter notebooks are provided for the prediction logic. It takes the input data set and create the required model file called “XXX.pkl”. XXX denotes the output file related to the kind of model used for this purpose. It also creates the scaler file gives the details about how the inputs are scaled before fitting the models.
2.	Use the prediction model
Using the model and scaler file (created by the Jupyter notebook) predict the house prices from various models used.
Implement the code :
Jupyter Notebook:
1. Load the data from bank.csv
2. Preprocess the data by removing the missing values, null values.
3. Find correlation between the features for the selection of features.
4. split the data train vs test 70-30 ratio.
5. It doesnot do any regularization

6.  Another deviation is to perform various modelling algorithms like Linear regression, Logistic Regression,  Random Forest Regression, SVR and ADA Boost, XG boost Regressions. Similar ones are grouped in one notebook.
7. Save the pickle file and the scaler files for the client to use for the predictions from user inputs.
8. The client application is created using streamlit and the requirement file with the details in the session Installation.
9. How to make it work::

	Copy following files xx.pkl, scaler file xx.save, homepredictor.py.
eg. VK_linR_Model_sm.pkl linR = Linear Regression, sm = small input
Command Prompt and Python file:
Refer Usage section of this document how to use client application.

Output
HTML page using the local host. (A typical output)



Enhancements:
The client application will be upgraded for publishing the application on Heroku server website. In that case we don’t need to copy or download the files from the GitHub.
Contact:
Any questions Please contact Vijayakumar Kurup
