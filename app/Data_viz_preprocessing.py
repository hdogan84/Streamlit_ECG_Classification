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
from functions import calculate_average_values #just for debugging, remove later.


def Data_viz_preprocessing():
    
    data_path = "C:/Users/dgnhk/dst_project/heartbeat_data"
    #data_path = "/home/simon/Datascientest_Heartbeat/jan24_bds_int_heartbeat/data/KAGGLE_datasets/heartbeat"

    ### Create Title
    st.title("Datavizualization and Preprocessing")
    st.write("As shown in the Introduction page the raw data from Kaggle was already preprocessed and segmented well enough to start directly with data exploration. No extensive preprocessing steps had to be performed.")
    st.header("Datavizualization")
    st.subheader("Downloading the Datasets from Kaggle")
    st.write("We implemented a handy function to directly download the datasets from Kaggle:")
    # Code collapsible section
    show_download_code_Kaggle() #calls the function to show the downloaded code since this takes large volume of code...
    st.write("All Datasets are stored in a local folder, since they are too big to be pushed onto github (>100 mb).")
    mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace(data_path)
    
    ### Showing code
    st.text("importing datasets in our workspace with the following command: ")
    with st.echo(): 
            mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace(data_path)

    #Checkbox for the button
    is_button_enabled = st.checkbox("Show random plotting")
    if is_button_enabled:
          if st.button("Plot random row from random Dataset"):
            plot_random_row(data_path)


    #debugging the calculate averages function --> Can be used for further functions with selections of one dataset.
    is_check_output_avg_values = st.checkbox("Test the outpout of the calculate average values function:")
    if is_check_output_avg_values:
         debug_avg_values = calculate_average_values(dataset_folder = data_path,
                                                     dataset_names = ["mitbih_test.csv", "mitbih_train.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"])
         selected_key = st.selectbox("Select a dataset key:", options=list(debug_avg_values.keys()))
         st.write(f"Key: {selected_key}")
         st.write(debug_avg_values[selected_key])
         for i, class_avg in enumerate(debug_avg_values[selected_key]):
            #class_label = disease_names_mitbih[i]
            plt.plot(class_avg, linestyle='--', alpha=0.8) #, label=f"{class_label} Average", , color=color_palette[i],
         st.pyplot()

    from PIL import Image
    st.write("Mean values of time signals in MITBIH Train set")
    # open an image
    img = Image.open("../assets/Report1_Fig4_Mitbih_train.png")
    # Plot the image
    st.image(img)

    st.write("Mean values of time signals in MITBIH Test set")
    # open an image
    img2 = Image.open("../assets/Report1_Fig4_Mitbih_test.png")
    # Plot the image
    st.image(img2)

         
