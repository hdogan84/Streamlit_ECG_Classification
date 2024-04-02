import streamlit as st
import base64
import pandas as pd 
import seaborn as sns 
import streamlit as st 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle
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
def load_pkl_model(model_path_pkl):
    pkl_file = open(model_path_pkl, 'rb')
    model = pickle.load(pkl_file)
    pkl_file.close()
    return model

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


@st.cache_data
def calculate_average_values(dataset_folder, dataset_names):
    """
    Calculates average values for each class in each dataset.
    Is used in the plot_random_row function --> The calculation of the mean values is always the same and must not be repeated!
    Args:
    - dataset_folder (str): Path to the folder containing the datasets.
    - dataset_names (list of str): List of dataset names.

    Returns:
    - average_values (dict): Dictionary containing average values for each class in each dataset.
    """
    average_values = {}
    for dataset_name in dataset_names:
        df = pd.read_csv(f"{dataset_folder}/{dataset_name}")
        df_columns = df.columns.tolist()
        target_column = df_columns[-1]
        average_values[dataset_name] = df.groupby(target_column).mean().values
    return average_values

#@st.cache_data is not necessary, because this function is always called with random row numbers?
def plot_random_row(dataset_folder = "/home/simon/Datascientest_Heartbeat/jan24_bds_int_heartbeat/data/KAGGLE_datasets/heartbeat", dataset_names = ["mitbih_test.csv", "mitbih_train.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"]):
    """
    Plots a random row from one of the datasets.

    Args:
    - dataset_folder (str): Path to the folder containing the datasets.
    - dataset_names (list of str): List of dataset names.
    """
    # Load average values
    average_values = calculate_average_values(dataset_folder, dataset_names)
    #dictionary for classes --> This should be a globally available dict, so that it cannot be changed from somewhere else.
    disease_names_mitbih = {0: "Normal", 1: "Supraventricular", 2: "Ventricular", 3: "Fusion V and N", 4: "Paced / Fusion Paced and Normal / Unknown"}
    disease_names_ptbdb = {0: "Normal", 1: "Abnormal"}
    # Select a random dataset
    selected_dataset = np.random.choice(dataset_names)

    # Load the dataset
    dataset_path = f"{dataset_folder}/{selected_dataset}"
    df = pd.read_csv(dataset_path)

    # Define custom color palette for dashed lines
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Select a random row
    random_row_index = np.random.randint(len(df))
    random_row = df.iloc[random_row_index]

    # Plot the random row
    plt.figure(figsize=(10, 6))
    plt.plot(random_row[:-1], label="Random Row")  # Exclude the last column (target)
     # Plot average values for each class in the background with dashed lines
    if "ptbdb" in selected_dataset:
        ptbdb_datasets = {"ptbdb_normal.csv": "Normal", "ptbdb_abnormal.csv": "Abnormal"}
        for i, (dataset, class_label) in enumerate(ptbdb_datasets.items()):
            class_avg = average_values[dataset][0] #this is the key to make it work --> THe output list was doubled [[]], i don´t exactly know why
            plt.plot(class_avg, linestyle='--', color=color_palette[i], alpha=0.8, label=f"{class_label} Average")
    else:
        for i, class_avg in enumerate(average_values[selected_dataset]):
            class_label = disease_names_mitbih[i]
            plt.plot(class_avg, linestyle='--', color=color_palette[i], alpha=0.8, label=f"{class_label} Average")

    plt.title(f"Random Row ({random_row_index}) from {selected_dataset} Dataset")
    plt.xlabel("Timesteps in ms")
    plt.ylabel("Normalized ECG Amplitude")
    plt.grid(True)
    x_ticks_positions = np.arange(0, 188, 10) #Positions of the x-ticks (we choose an tick for every 10th position)
    x_tick_labels = [i * 8 for i in x_ticks_positions] #each timestep is 8ms long, already multiplied by 10 due to x_ticks_positions..
    plt.xticks(x_ticks_positions, x_tick_labels) #we can now change the x_ticks accordingly.
    plt.legend()
    st.pyplot()

    # Show the target column with disease names
    dataset_disease_names = disease_names_mitbih if selected_dataset == "MITBIH" else disease_names_ptbdb
    target_value = random_row.iloc[-1]
    heartbeat_class = dataset_disease_names.get(target_value, "Unknown")

    st.write("Classification of Heartbeat:", heartbeat_class)