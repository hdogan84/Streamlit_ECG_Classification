import streamlit as st
#from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow as tf

#@st.cache_data #cannot be properly used here
def run():
    st.header("Model Architectures")

    dataset_name = st.selectbox("Select Dataset on which the model was trained", ["MITBIH", "PTBDB"])
    
    model_options = ["Simple_ANN", "Simple_CNN", "Advanced_CNN"]
    selected_model = st.radio("Select the DL model", options=model_options)
    
    if dataset_name == "MITBIH":
        num_classes = 5 
    elif dataset_name == "PTBDB":
        num_classes = 2

    if selected_model == 'Simple_ANN':
        model = build_simple_ann(num_classes)
    elif selected_model == 'Simple_CNN':
        model = build_simple_cnn(num_classes)
    elif selected_model == 'Advanced_CNN':
        model = build_adv_cnn(num_classes)
    else: 
        st.write("No models selected")

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    st.text(model_summary)

    st.header("Experiment Design")
    st.write("We hereby present shortly an overview on the experiment design for all DL Models.")
    data = {
    'Experiment Design': [f'Experiment {i}' for i in range(1, 5)],
    'Model Type': ['All Models'] * 4,
    'Epochs': [70, 70, 70, 70],
    'batch_size': [10, 10, 10, 10],
    'patience': [10, 10, 70, 70],
    'Initial_Learning_Rate': [0.001, 0.0001, 0.0001, 0.001],
    'lr_reduction_rate': ['N/A', 'N/A', 'N/A', '0.5 @10 Epochs'],
    'Additional_Info': ['', '', 'Early Stop Callback is essentially deactivated.', 'Early Stop Callback is essentially deactivated. Learning rate is halved each 10 epochs, initial lr is used.']
    }
    df = pd.DataFrame(data)
    #st.table(df) #works, but not pretty
    st.write(
    df.style
    .set_properties(**{'text-align': 'left'})
    .set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])
    .set_caption("Experiment Details")
    )
    
            
        

#here the actual function is called (from  app.py)
def Page_DL_Stage_1():
    run()


####### Here build model functions are defined intern, WITHOUT load_weights, so that 
####### it works on all laptops

# build advanced CNN model 
@st.cache_data
def build_adv_cnn(num_classes=5):
    """
    builds the advanced CNN model from the reports
    """

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
    adv_cnn_model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) #softmax classes are dynamically adjusted according to the dataset!
    
    return adv_cnn_model

@st.cache_data
def build_simple_cnn(num_classes=5):
    """
    builds the simple CNN model from the reports
    """

    class Config_CNN:
        Conv1_filter_num = 32
        Conv1_filter_size = 3
        

    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.Conv1D(Config_CNN.Conv1_filter_num, Config_CNN.Conv1_filter_size, activation='relu', input_shape=(187, 1))) # We add one Conv1D layer to the model
    cnn_model.add(tf.keras.layers.Flatten()) # After 
    cnn_model.add(tf.keras.layers.Dense(60, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    cnn_model.add(tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    cnn_model.add(tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    cnn_model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) #softmax classes are dynamically adjusted according to the dataset!
    
    return cnn_model

@st.cache_data
def build_simple_ann(num_classes=5):
    """
    builds the simple ANN model from the reports
    """

    ann_model = tf.keras.models.Sequential()
    ann_model.add(tf.keras.layers.Dense(60, activation=tf.keras.layers.LeakyReLU(alpha=0.001), input_shape=(187,)))
    ann_model.add(tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    ann_model.add(tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    ann_model.add(tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    ann_model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) #softmax classes are dynamically adjusted according to the dataset!
    
    return ann_model

