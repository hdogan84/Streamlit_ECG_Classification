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
from functions import train_model, download_datasets, show_download_code_Kaggle, load_datasets_in_workingspace


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


    """### Showing the data
    if st.checkbox("Showing the data") :
        line_to_plot = st.slider("select le number of lines to show", min_value=3, max_value=df.shape[0])
        st.dataframe(df.head(line_to_plot))

    if st.checkbox("Missing values") : 
        st.dataframe(df.isna().sum())


    ### Preprocessing 

    # Drop Missing values
    df = df.dropna()

    # Drop some columns
    df = df.drop(['sex', 'title', 'cabin', 'embarked'], axis = 1)

    # Select the target
    y = df['survived']

    # Select the features
    X = df.drop('survived', axis =1 )

    # select how to split the data
    train_size = st.sidebar.slider(label = "Choix de la taille de l'échantilllon de train", min_value = 0.2, max_value = 1.0, step = 0.05)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size)


    ### Display graph
    st.text('Class distribution with seaborn')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.countplot(df['pclass'])
    st.pyplot()

    ### Classification 

    #  Baseline model
    model = LogisticRegression() 

    # Model training
    model.fit(X_train, y_train)

    # Benchmark Model evaluation
    st.write("Logisitic regression accuracy (This is my Benchmark):" , model.score(X_test,y_test))

    # Other models
    model_list = ['Decision Tree', 'KNN']
    model_choisi = st.selectbox(label = "Select a model" , options = model_list)


    # Showing the accuracy for the orthers models (for comparison)
    st.write("Accuracy for some models for comparison: ")
    st.write("Score test", train_model(model_choisi, X_train, y_train, X_test, y_test))"""