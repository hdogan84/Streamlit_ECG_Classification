# Core Pkg
import pandas as pd 
import seaborn as sns 
import streamlit as st 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from functions import train_model, download_datasets, show_download_code_Kaggle, load_datasets_in_workingspace, plot_random_row
from functions import load_pkl_model, predict_with_ML 
from sklearn.metrics import accuracy_score, classification_report


def Page_ML_Stage_1(data_path = "../data/heartbeat"):
    """
    Function to display the page for the modeling with the ML models.
    - data_path = path where the datasets mitbih and ptbdb are stored (normally ../data/heartbeat)

    """

    ### Create Title
    st.title("Predicting with ML")

    #this should be done in the second page, and also ptbdb data needs to be concated and shuffled --> also shown in page 2
    mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace(data_path)
    
        #Checkbox for the button
    st.subheader("MITBIH Predictions")
    #we now only call the designated prediction function!
    predict_with_ML(mitbih_test)

    st.subheader(":red[PTBDB Predictions (on ptbdb normal)]")
    #we now only call the designated prediction function!
    predict_with_ML(ptbdb_normal)

    st.subheader(":red[PTBDB Predictions (on ptbdb abnormal)]")
    #we now only call the designated prediction function!
    predict_with_ML(ptbdb_abnormal)

    st.header(":red[Notes for further Improvement:]")
    st.write("Make a selection routine instead of predefined tables:")
    st.write("- Select Dataset")
    st.write("- select the ML Models that should be compared together")
    st.write("- Select the comparison Method:")
    st.write("- - Single Row: Use a single row (random) to predict --> Compare the real class with the predicted classes of the selected models.")
    st.write("- - Complete Dataset: Print classification reports, confusion matrix, bar plots with metrics for each model selected or find a way to show all results in one single plot (like results plot in the report)")
    st.write("- Print / plot the results")

         
