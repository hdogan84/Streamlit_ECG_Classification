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
    
    data_path = "../data/heartbeat" #using a relative path, if the path is not found, the data can be locally downlaoded and lands in a .gitignored folder.
    
    ### Create Title
    st.title("Data visualization and Preprocessing")
    st.write("As shown in the Introduction page the raw data from Kaggle was already preprocessed and segmented well enough to start directly with data exploration. No extensive preprocessing steps had to be performed.")
    st.header("Data visualization")
    st.subheader("Downloading the Datasets from Kaggle")
    st.write("We implemented a handy function to directly download the datasets from Kaggle:")
    # Code collapsible section
    show_download_code_Kaggle() #calls the function to show the downloaded code since this takes large volume of code...
    st.write("All Datasets are stored in a local folder, since they are too big to be pushed onto github (>100 mb).")
    if st.button("Push this button to download the kaggle datasets in a .gitignored folder on your computer to continue"):
         download_datasets(data_path)


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


#Simon: I am not sure why these pictures should be shown, when the random plot function does the same?
# Also: do not show only mitbih, but also ptbdb (test and train from mitbih are essentially the same.)
    from PIL import Image
    #st.subheader(":red[Mean Amplitude of Signals in the datasets (Is this necessary?)]")

    st.header("Data Resampling")

    st.subheader("MITBIH Dataset")
    img1 = Image.open("../assets/Report1_Fig11_Mitbih_resampling.bmp")
    # Plot the image
    st.image(img1)

    st.subheader("PTBDB Dataset")
    img2 = Image.open("../assets/Report1_Fig12_Ptbdb_resampling.png")
    # Plot the image
    st.image(img2)

    st.subheader("Outlier Detection (Mitbih Dataset)")
    img3 = Image.open("../assets/Report1_Fig7_Mitbih.png")
    # Plot the image
    st.image(img3)

    st.subheader("Correlation Matrix for Original data")

    comparison_method = st.radio("Select the Dataset", ["MITBIH", "PTBDB"])


    st.header(":red[Notes for further Improvement:]")
    st.write("- Usage of only Mitbih train and test as plots is not that useful. Use Mitbih test and ptbdb concateted")
    st.write("- Include function and show function code to generate test and train dataset from ptbdb. Also show over- and undersampling techniques and create datasets that are stored as variables for the app.")
    st.write("- Keep random plotting and maybe add selection switches")
    st.write("- Show a correlation matrix for selected datasets --> Can be copied from Notebook 1 essentially and modifided with selection switches")
    st.write("- Plot the outlier detection as function: Use sliders to configure how sensitive the outliers should be detected and show the result as hearbeat avg plot (difference between original values and with outliers removed)")
    st.write("- Basic statistics with pie plots and heartbeat shapes --> These can be shown generally and all functions (with sliders) above dynamically adjust these basic statistics plots?")
    """
    This function is not used in the final presentation, but could serve as an example on how to select our datasets.
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
         st.pyplot()"""