import streamlit as st
import base64
import pandas as pd 
import seaborn as sns 
import streamlit as st 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
from kaggle.api.kaggle_api_extended import KaggleApi



def displayPDF(file, width=1000, height=700, caption=None):
    """
    Takes a pdf file as argument and shows it as st.markdown directly.
    The Argument <<file>> must therefore be also a direct path to the file.
    """
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" type="application/pdf"></iframe>'
    
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

    # Displaying Caption
    if caption:
        st.write(f'<p style="font-size: small; color: #888888;">{caption}</p>', unsafe_allow_html=True)

    
# Custom function to train models --> Can be removed later, but serves as an example for now
# st.cache is used to load the function into memory, i.e. it will be checked, if the function has been called with the specific parameters already.
@st.cache_data
def train_model(model_choisi, X_train, y_train, X_test, y_test) :
    if model_choisi == 'Regression Logisitic' : 
        model = LogisticRegression()
    elif model_choisi == 'Decision Tree' : 
        model = DecisionTreeClassifier()
    elif model_choisi == 'KNN' : 
        model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score

@st.cache_data
def download_datasets(download_path, dataset_owner="shayanfazeli", dataset_name="heartbeat"):
    """
    Downloads datasets from Kaggle API and stores them in the specified folder.

    Args:
    - dataset_owner (str): Owner of the dataset on Kaggle.
    - dataset_name (str): Name of the dataset on Kaggle.
    - download_path (str): Path to the download folder outside the Git repository.
    """
    # Configure and authenticate with the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Check if the dataset folder already exists
    if not os.path.exists(dataset_folder):
        # Dataset folder does not exist --> Download and save the datasets
        api.dataset_download_files(dataset_owner + "/" + dataset_name, path=os.path.join(download_path, dataset_name), unzip=True)
        print("Datasets are downloaded and unzipped.")
    else:
        # Dataset folder exists, but datasets might be missing
        missing_files = [] 
        for file_name in ["mitbih_test.csv", "mitbih_train.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"]:  
            file_path = os.path.join(dataset_folder, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)

        if missing_files:
            # If missing files are present, download ALL files and overwrite the old folder.
            api.dataset_download_files(dataset_owner + "/" + dataset_name, path=os.path.join(download_path, dataset_name), unzip=True, force=True)
            print("Missing data was downloaded and unzipped. All Datasets are now available.")
        else:
            print("All Datasets are already available.")

@st.cache_data
def show_download_code_Kaggle():
    with st.expander("Show download_datasets function code"):
            st.code(
                """
    import os
    from kaggle.api.kaggle_api_extended import KaggleApi

    def download_datasets(download_path, dataset_owner="shayanfazeli", dataset_name="heartbeat"):
        # Configure and authenticate with the Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Check if the dataset folder already exists
        dataset_folder = os.path.join(download_path, dataset_name)
        if not os.path.exists(dataset_folder):
            # Dataset folder does not exist --> Download and save the datasets
            api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True)
            print("Datasets are downloaded and unzipped.")
        else:
            # Dataset folder exists, but datasets might be missing
            missing_files = [] 
            for file_name in ["mitbih_test.csv", "mitbih_train.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"]:  
                file_path = os.path.join(dataset_folder, file_name)
                if not os.path.exists(file_path):
                    missing_files.append(file_name)

            if missing_files:
                # If missing files are present, download ALL files and overwrite the old folder.
                api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True, force=True)
                print("Missing data was downloaded and unzipped. All Datasets are now available.")
            else:
                print("All Datasets are already available.")
                """
            )

@st.cache_data #The path_to_datasets might be specified in a dynamic .env file?
def load_datasets_in_workingspace(path_to_datasets="/home/simon/Datascientest_Heartbeat/jan24_bds_int_heartbeat/data/KAGGLE_datasets/heartbeat"):
    #reading in the datasets from the local ../data folder --> this folder is not pushed on github and only locally available.
    mitbih_test = pd.read_csv(path_to_datasets + "/" + "mitbih_test.csv",header=None)
    mitbih_train = pd.read_csv(path_to_datasets + "/" + "mitbih_train.csv",header=None)
    ptbdb_abnormal = pd.read_csv(path_to_datasets + "/" + "ptbdb_abnormal.csv",header=None)
    ptbdb_normal = pd.read_csv(path_to_datasets + "/" + "ptbdb_normal.csv",header=None)
    return mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal