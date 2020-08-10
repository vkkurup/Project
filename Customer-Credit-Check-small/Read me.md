![](RackMultipart20200810-4-19zz667_html_6b795e7d87ea2be4.gif) ![](RackMultipart20200810-4-19zz667_html_2b1c981b86ab85bd.gif) ![](RackMultipart20200810-4-19zz667_html_f820581a29181e76.gif) ![](RackMultipart20200810-4-19zz667_html_32803752825085bd.gif) ![](RackMultipart20200810-4-19zz667_html_ac6784755a0b116b.gif) ![](RackMultipart20200810-4-19zz667_html_40b7ca959ea5caa8.gif)

Vijayakumar kurup

## **This document**

## **Project which predicts the Credit eligibility check using various ML algorithm and way to compare them.**

# Credit Elibility

# prediction

Python version

## Table of Contents:

#

Y

[Table of Contents: 1](#_Toc47904103)

[Introduction: 3](#_Toc47904104)

[Included with the project: 3](#_Toc47904105)

[Getting Started: 3](#_Toc47904106)

[Pre-requisites: 3](#_Toc47904107)

[Installation: 3](#_Toc47904108)

[Usage: 3](#_Toc47904109)

[Input File: 4](#_Toc47904110)

[Implement the code : 4](#_Toc47904111)

[**Jupyter Notebook:** 4](#_Toc47904112)

[**Command Prompt and Python file:** 4](#_Toc47904113)

[Output 4](#_Toc47904114)

[**HTML page using the local host. (A typical output)** 4](#_Toc47904115)

[5](#_Toc47904116)

[Enhancements: 5](#_Toc47904117)

[Contact: 5](#_Toc47904118)

###


## Introduction:

&quot;This Capstone project is created for predicting the credit eligibility for the customers. The dataset used for this project is a comma separated file called &quot;Bank.csv with&quot;. Several ML algorithms were used in doing this.

### Included with the project:

This project includes the input dataset, the Jupyter notebook, the necessary python packages, and a client implementation.

## Getting Started:

The data set has the following features

&#39;age&#39;, &#39;balance&#39;, &#39;duration&#39;, &#39;pdays&#39;, &#39;previous&#39;, &#39;poutcome&#39;, &#39;y&#39;,

&#39;jobcat&#39;, &#39;educationcat&#39;, &#39;defaultcat&#39;, &#39;housingcat&#39;.

.

## Pre-requisites:

To run this project and evaluating various algorithms, some basic computer knowledge along with knowledge of running a python project, and knowledge of prediction logics and HTML client application.

## Installation:

Install the following packages

Python 3.7

streamlit

pandas

joblib

sklearn

or use the requirement.txt for the installation if IDE like PyCharm is used.

## Usage:

From any command prompt like conda with environment where all the above packages given in the installation section, are installed and then run the following command

&quot;streamlit run credit-small.py&quot;

### Input File:

There are two aspects of this project.

1. Create the prediction model.

To do this Jupyter notebooks are provided for the prediction logic. It takes the input data set and create the required model file called &quot;XXX.pkl&quot;. XXX denotes the output file related to the kind of model used for this purpose. It also creates the scaler file gives the details about how the inputs are scaled before fitting the models.

1. Use the prediction model

Using the model and scaler file (created by the Jupyter notebook) predict the house prices from various models used.

### Implement the code :

**Jupyter Notebook:**

1. Load the data from bank.csv

2. Preprocess the data by removing the missing values, null values.

3. Find correlation between the features for the selection of features.

4. split the data train vs test 70-30 ratio.

5. It doesnot do any regularization

6. Another deviation is to perform various modelling algorithms like Linear regression, Logistic Regression, Random Forest Regression, SVR and ADA Boost, XG boost Regressions. Similar ones are grouped in one notebook.

7. Save the pickle file and the scaler files for the client to use for the predictions from user inputs.

8. The client application is created using streamlit and the requirement file with the details in the session Installation.

9. How to make it work::

Copy following files xx.pkl, scaler file xx.save, homepredictor.py.

eg. VK\_linR\_Model\_sm.pkl linR = Linear Regression, sm = small input

**Command Prompt and Python file:**

Refer Usage section of this document how to use client application.

### Output

**HTML page using the local host. (A typical output)**

![](RackMultipart20200810-4-19zz667_html_f525726726c2acf9.png)

## Enhancements:

The client application will be upgraded for publishing the application on Heroku server website. In that case we don&#39;t need to copy or download the files from the GitHub.

## Contact:

Any questions Please contact [Vijayakumar Kurup](mailto:vkkurup@gmail.com)