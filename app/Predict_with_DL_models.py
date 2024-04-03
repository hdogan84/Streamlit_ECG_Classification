# Core Pkg
import streamlit as st
from functions import load_datasets_in_workingspace, predict_with_DL
from sklearn.metrics import classification_report


def Page_DL_Stage_2(data_path = "../data/heartbeat"):
    
       ### Create Title
    st.title("Predicting with DL")
    mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace(data_path) #this could be done globally? or as argument?
    
    #y_test = mitbih_test[187]
    #X_test = mitbih_test.drop(187,axis=1)

    #Checkbox for the button
    st.subheader("MITBIH predictions with Advanced CNN on test set")
    predict_with_DL(mitbih_test, model="advanced_cnn", model_path="../assets/experiment_4_MITBIH_A_Original.weights.h5")


    st.header(":red[Notes for further Improvement:]")
    st.write("Make a selection routine instead of predefined tables:")
    st.write("- Select Dataset")
    st.write("- select the ML Models that should be compared together")
    st.write("- Select the comparison Method:")
    st.write("- - Single Row: Use a single row (random) to predict --> Compare the real class with the predicted classes of the selected models.")
    st.write("- - Complete Dataset: Print classification reports, confusion matrix, bar plots with metrics for each model selected or find a way to show all results in one single plot (like results plot in the report)")
    st.write("- Print / plot the results")




    

         
