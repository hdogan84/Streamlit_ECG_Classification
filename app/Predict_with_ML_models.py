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
from functions import load_pkl_model 
from sklearn.metrics import accuracy_score, classification_report


def predict_with_ML():
    
    data_path = "C:/Users/dgnhk/dst_project/heartbeat_data"
    #data_path = "/home/simon/Datascientest_Heartbeat/jan24_bds_int_heartbeat/data/KAGGLE_datasets/heartbeat"
     #RF Classifier model pickle filepath
    rfc_path = "../assets/RFC_Optimized_Model_with_Gridsearch_MITBIH_A_Original.pkl"

    ### Create Title
    st.title("Predicting with ML")
    st.write("As shown ")
    # Code collapsible section
    show_download_code_Kaggle() #calls the function to show the downloaded code since this takes large volume of code...
    st.write("All Datasets are stored in a local folder, since they are too big to be pushed onto github (>100 mb).")
    mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace(data_path)
    
    y_test = mitbih_test[187]
    X_test = mitbih_test.drop(187,axis=1)


    #Checkbox for the button
    st.subheader("MITBIH preedictions")
    model = load_pkl_model(rfc_path)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, digits=4)
    st.text(report)

    from PIL import Image
    st.write("Confusion Matrix as a picture could be good here")
    

    

         
