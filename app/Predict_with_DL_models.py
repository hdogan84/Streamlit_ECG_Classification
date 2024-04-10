import streamlit as st
from functions import load_datasets_in_workingspace, generate_results, display_classification_report, display_confusion_matrix, display_radar_charts, display_lineplot
#from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

#@st.cache_data #cannot be properly used here
def run():
    st.title("Predicting with Deep learning (DL) models")

    dataset_names = st.multiselect("Select Datasets (more than one option possible)", ["MITBIH", "PTBDB"])
    selected_sampling = st.multiselect("Select the sampling method for the trainingset on which the models were trained",
                                     ["A_Original", "B_SMOTE"])
    model_options = ["Simple_ANN", "Simple_CNN", "Advanced_CNN"]
    selected_models = st.multiselect("Select the DL models (more than one option possible)", options=model_options)
    experiment_options = ["1", "2", "3", "4"]
    selected_experiments = st.multiselect("Select the experiments the models were trained on (more than one option possible)",
                                          options=experiment_options)
    comparison_method = st.radio("Select the comparison method", ["Single Row (Random)", "Complete Dataset"])

    if st.button("Generate Results"):
        st.session_state["all_results"] = {}
        all_results = st.session_state['all_results']
        
        for dataset_name in dataset_names:
            #st.subheader(f"Results for {dataset_name} Dataset") #this is confusing since no results are displayed with this function --> Should be called when the actual results are presented.
            mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace()
            dataset_to_use = mitbih_test if dataset_name == "MITBIH" else pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1, random_state=42)
            num_classes = 5 if dataset_name == "MITBIH" else 2
            results = generate_results(dataset_name, dataset_to_use, selected_models,
                                                                          selected_experiments, comparison_method, selected_sampling, num_classes)
            all_results[dataset_name] = results
            st.session_state['all_results'] = all_results #otherwise everything will be lost...
           
        #debugging how the results dictionary looks like...
        #st.write("How does the all_results dictionary look like?")
        #st.write(all_results)
        st.info("The results have been generated, continuing with showing the results is possible.")
    
    #display options should be shown after the results have been generated.
    display_options = st.multiselect("Select Display Option(s)",
                                     ["Classification Report", "Confusion Matrix", "Line Plot (Metrics)"], default =[]) #, "Radar Chart (Debugging)"
    
    if st.button("Show Results"):
        st.header("Results")
        all_results = st.session_state["all_results"] #since all_results is not put trough all buttons, it is newly created trough the persistend session_state variable...
               
        if "Classification Report" in display_options:
            st.subheader("Classification Report")
            display_classification_report(all_results)

        if "Confusion Matrix" in display_options:
            st.subheader("Confusion Matrices")
            display_confusion_matrix(all_results)

        if "Line Plot (Metrics)" in display_options:
            st.subheader("Line Plot(s) for the different metrics")
            display_lineplot(all_results)

        if "Radar Chart (Debugging)" in display_options:
            st.subheader("Radar Chart (Debugging)")
            display_radar_charts(all_results)

#here the actual function is called (from  app.py)
def Page_DL_Stage_2():
    run()
