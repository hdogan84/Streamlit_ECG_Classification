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
from functions import train_model, download_datasets, show_download_code_Kaggle, load_datasets_in_workingspace, plot_random_row, plot_with_outliers, plot_correlation_matrix
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
    st.image(img1, caption="Fig 6: Original (left) and resampled (middle, right) distribution of classes for the MITBIH dataset. Test and train set are distributed in the same way.")

    st.subheader("PTBDB Dataset")
    img2 = Image.open("../assets/Report1_Fig12_Ptbdb_resampling.png")
    # Plot the image
    st.image(img2, caption="Fig 7: Original (left) and resampled (middle, right) distribution of classes for the PTBDB dataset after concatening.")

    st.header("Outlier Detection")
    st.subheader("First evaluation with boxplots (Mitbih)")
    img3 = Image.open("../assets/Report1_Fig7_Mitbih.png")
    # Plot the image
    st.image(img3, caption="Fig 8: Outlier detection with box plot for the MITBIH Dataset and Normal (N) class. Zoomed subplot shows the details.)")

    st.subheader("Further Evaluation with quantile based removal of outliers") 
    st.write("To further evaluate how outliers would influence the heartbeat shape and thus possibly the results of an examination (either manually or with machine learning) we show the effect of outliers on the overall shape of an heartbeat.")
    
    dataset_selected = st.radio("Select the Dataset", ["MITBIH (Trainset)", "PTBDB (Concated set)"])
    lower_quantile = st.slider("Lower Quantile", min_value=0.0, max_value=0.5, value=0.25, step=0.05)
    upper_quantile = st.slider("Upper Quantile", min_value=0.5, max_value=1.0, value=0.75, step=0.05)
    if dataset_selected == "MITBIH (Trainset)":
         plot_with_outliers(mitbih_train, lower_quantile=lower_quantile, upper_quantile=upper_quantile)
    if dataset_selected == "PTBDB (Concated set)":
         data =  pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1, random_state=42) #short concatening
         plot_with_outliers(data, lower_quantile=lower_quantile, upper_quantile=upper_quantile)


    st.header("Correlation Matrix for Original data")

    comparison_method = st.radio("Select the Dataset", ["MITBIH (Trainset)", "PTBDB (Concated set)"], key="Selection for Correlation Matrix")
    if comparison_method == "MITBIH (Trainset)":
         plot_correlation_matrix(mitbih_train, selected_dataset="MITBIH (Train)")
    if comparison_method == "PTBDB (Concated set)":
         data =  pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1, random_state=42) #short concatening
         plot_correlation_matrix(data, selected_dataset="PTBDB (concatenated)")

