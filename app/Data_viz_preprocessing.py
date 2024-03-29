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


def Data_viz_preprocessing():

    ### Create Title
    st.title("Datavizualization and Preprocessing")
    st.write("As shown in the Introduction page the raw data from Kaggle was already preprocessed and segmented well enough to start directly with data exploration. No extensive preprocessing steps had to be performed.")
    st.header("Datavizualization")
    st.subheader("Downloading the Datasets from Kaggle")
    st.write("We implemented a handy function to directly download the datasets from Kaggle:")
    # Code collapsible section
    show_download_code_Kaggle() #calls the function to show the downloaded code since this takes large volume of code...
    st.write("All Datasets are stored in a local folder, since they are too big to be pushed onto github (>100 mb).")
    mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace()
    
    ### Showing code
    st.text("importing datasets in our workspace with the following command: ")
    with st.echo(): 
            mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace()

    #Checkbox for the button
    is_button_enabled = st.checkbox("Show random plotting")
    if is_button_enabled:
          if st.button("Plot random row from random Dataset"):
            plot_random_row()