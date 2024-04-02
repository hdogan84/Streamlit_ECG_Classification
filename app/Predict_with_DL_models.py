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
import tensorflow as tf


def predict_with_DL():
    
    data_path = "C:/Users/dgnhk/dst_project/heartbeat_data"
    #data_path = "/home/simon/Datascientest_Heartbeat/jan24_bds_int_heartbeat/data/KAGGLE_datasets/heartbeat"
    #RF Classifier model pickle filepath
    model_path = "../assets/experiment_4_MITBIH_A_Original.weights.h5"

    ### Create Title
    st.title("Predicting with DL")
    mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace(data_path)
    
    y_test = mitbih_test[187]
    X_test = mitbih_test.drop(187,axis=1)

    #Checkbox for the button
    st.subheader("MITBIH predictions")
    model = build_model_adv_cnn(model_path)
    print(model.summary())

    #model.load_weights("../assets/experiment_4_MITBIH_A_Original.weights.h5")

    predictions = model.predict(X_test).argmax(axis=1)
    print(predictions.shape)
    report = classification_report(y_test, predictions, digits=4)
    st.dataframe(report)

    from PIL import Image
    st.write("Confusion Matrix as a picture could be good here")

# build model and load weights from h5 file
def build_model_adv_cnn(model_filepath):

    class Config_Advanced_CNN:
        Conv1_filter_num = 32
        Conv1_filter_size = 3
        

    adv_cnn_model = tf.keras.models.Sequential()
    adv_cnn_model.add(tf.keras.layers.Conv1D(Config_Advanced_CNN.Conv1_filter_num, Config_Advanced_CNN.Conv1_filter_size, activation='relu', input_shape=(187, 1))) # We add one Conv1D layer to the model
    adv_cnn_model.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)) # We add one Conv1D layer to the model
    adv_cnn_model.add(tf.keras.layers.Conv1D(Config_Advanced_CNN.Conv1_filter_num//2, Config_Advanced_CNN.Conv1_filter_size, activation='relu' )) # We add one Conv1D layer to the model
    adv_cnn_model.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)) # We add one Conv1D layer to the model
    adv_cnn_model.add(tf.keras.layers.Flatten()) # After  
    adv_cnn_model.add(tf.keras.layers.Dropout(rate=0.2))
    adv_cnn_model.add(tf.keras.layers.Dense(120, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    adv_cnn_model.add(tf.keras.layers.Dense(60, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    adv_cnn_model.add(tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    adv_cnn_model.add(tf.keras.layers.Dense(5, activation='softmax'))

    #print(adv_cnn_model.summary())
    
    #adv_cnn_model.load_weights("../assets/experiment_4_MITBIH_A_Original.weights.h5")
    #adv_cnn_model.load_weights("/kaggle/input/ecg-cnn-bestmodels/experiment_4_MITBIH_A_Original.weights.h5")

    return adv_cnn_model



    

         
