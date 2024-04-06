import streamlit as st
from functions import load_datasets_in_workingspace, predict_with_DL
from sklearn.metrics import classification_report, confusion_matrix
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

    display_options = st.multiselect("Select Display Option(s)",
                                     ["Classification Report", "Confusion Matrix"],
                                     default=[])

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
    if st.button("Show Results"):
        st.header("Results")
        all_results = st.session_state["all_results"] #since all_results is not put trough all buttons, it is newly created trough the persistend session_state variable...
               
        if "Classification Report" in display_options:
            st.subheader("Classification Report")
            display_classification_report(all_results)

        if "Confusion Matrix" in display_options:
            st.subheader("Confusion Matrices")
            display_confusion_matrix(all_results)

#@st.cache_data #this function must not be cached, otherwise the randomness of the single row is lost?
def generate_results(dataset_name, dataset_to_use, selected_models, selected_experiments,
                     comparison_method, selected_sampling, num_classes):
    results = {}
    row_index = np.random.randint(len(dataset_to_use)) #only used when comparison_method == "Single Row (Random)", but if not defined here, a new row index will be selected for each model.
    single_row = pd.DataFrame(dataset_to_use.iloc[row_index].values.reshape(1, -1))
    

    for model_name in selected_models:
        results[model_name] = {} #initializing empty dictionary for each model_name (this is the same strategy as with all_results dictionary!)
        for sampling_method in selected_sampling:
            results[model_name][sampling_method] = {}
            for experiment in selected_experiments:
                model_path = f"../assets/DL_Models/{model_name}/experiment_{experiment}_{dataset_name}_{sampling_method}.weights.h5"
                if comparison_method == "Single Row (Random)":
                    
                    y_true = dataset_to_use.iloc[row_index, 187]
                    prediction, report = predict_with_DL(test=single_row, model=model_name, model_path=model_path,
                                                        show_conf_matr=False, num_classes=num_classes)
                    results[model_name][sampling_method]["Experiment " + experiment] = {
                        "y_true": y_true,
                        "prediction": int(prediction[0]),
                        "report": report
                    }

                elif comparison_method == "Complete Dataset":
                    prediction, report = predict_with_DL(test=dataset_to_use, model=model_name, model_path=model_path,
                                                        show_conf_matr=False, num_classes=num_classes)
                    y_true = dataset_to_use[187].values
                    results[model_name][sampling_method]["Experiment " + experiment] = {
                        "y_true": y_true,
                        "prediction": prediction, #this could make trouble, but lets see
                        "report": report
                    }
    
    return results


@st.cache_data
def display_classification_report(results):
    for dataset_name, dataset_results in results.items():
        st.subheader(f"Dataset {dataset_name}")
        for model_name, model_result in dataset_results.items():
            st.subheader(f"Model {model_name}")
            for sampling_method, sampling_method_result in model_result.items():
                st.subheader(f"Sampling Method training Set: {sampling_method}")
                for experiment, experiment_result in sampling_method_result.items():
                    st.subheader(f"{experiment}") #this is already correctly named due to new dictionary structure.
                    st.write("Classification Report:")
                    if isinstance(experiment_result, dict) and "report" in experiment_result:
                        report = experiment_result["report"]
                        st.dataframe(report)


@st.cache_data
def display_confusion_matrix(results):
    for dataset_name, dataset_results in results.items():
        st.subheader(f"Dataset {dataset_name}")
        for model_name, model_result in dataset_results.items():
            st.subheader(f"Model {model_name}")
            for sampling_method, sampling_method_result in model_result.items():
                st.subheader(f"Sampling Method training Set: {sampling_method}")
                for experiment, experiment_result in sampling_method_result.items():
                    st.subheader(f"{experiment}") #this is already correctly named due to new dictionary structure.
                    st.write("Confusion Matrix:")
                    y_true = experiment_result["y_true"]
                    prediction = experiment_result["prediction"]
                    cm = confusion_matrix(y_true, prediction)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
                    ax.set_title(f"Confusion Matrix for {model_name} in {experiment} on {dataset_name} ({sampling_method})")
                    ax.set_xlabel('Predicted Labels')
                    ax.set_ylabel('True Labels')
                    st.pyplot(fig)


#here the actual function is called (from  app.py)
def Page_DL_Stage_2():
    run()
