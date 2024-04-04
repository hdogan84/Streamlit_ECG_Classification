import streamlit as st
from functions import load_datasets_in_workingspace, predict_with_DL
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


class DLModelPredictor:
    def __init__(self):
        self.data_path = "../data/heartbeat"
        # initializing all_results from the session state if given, else this will be an empty dictionary
        if 'all_results' not in st.session_state:
            st.session_state['all_results'] = {}
        self.all_results = st.session_state['all_results']
        # initializing display_options from the session state if given, else this will be an empty list
        if 'display_options' not in st.session_state:
            st.session_state['display_options'] = []
        self.display_options = st.session_state['display_options']
        # apparently the init is run each time something is pressed... that's why self.all_results is working throughout the code... Not chic, but working.

    def generate_results_and_display(self, dataset_name, dataset_to_use, selected_models, selected_experiments,
                                     comparison_method, selected_sampling, num_classes):
        results = {}

        for model_name in selected_models:
            for experiment in selected_experiments:
                model_path = f"../assets/DL_Models/{model_name}/experiment_{experiment}_{dataset_name}_{selected_sampling}.weights.h5"

                if comparison_method == "Single Row (Random)":
                    row_index = np.random.randint(len(dataset_to_use))
                    single_row = pd.DataFrame(dataset_to_use.iloc[row_index].values.reshape(1, -1))
                    y_true = dataset_to_use.iloc[row_index, 187]
                    prediction, report = predict_with_DL(test=single_row, model=model_name, model_path=model_path,
                                                         show_conf_matr=False, num_classes=num_classes)
                    results[f"{model_name}_Exp{experiment}"] = report
                    st.dataframe(report)

                elif comparison_method == "Complete Dataset":
                    prediction, report = predict_with_DL(test=dataset_to_use, model=model_name, model_path=model_path,
                                                         show_conf_matr=False, num_classes=num_classes)
                    y_true = dataset_to_use[187].values
                    results[f"{model_name}_Exp{experiment}"] = report
                    st.dataframe(report)

        if comparison_method == "Complete Dataset":
            for result_key, result in results.items():
                cm = confusion_matrix(y_true, prediction)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
                ax.set_title(f"Confusion Matrix for {result_key} and Dataset {dataset_name}_{selected_sampling}")
                ax.set_xlabel('Predicted Labels')
                ax.set_ylabel('True Labels')
                st.pyplot(fig)

        # Now, update the all_results session state
        self.all_results[dataset_name] = results
        st.session_state['all_results'] = self.all_results
        st.write("Here comes the debugging for the generated results....")
        st.write(self.all_results)

        # Display classification report if selected
        if "Classification Report" in self.display_options:
            st.write("Displaying Classification Report...")
            self.display_classification_report(results)

    def display_classification_report(self, results):
        for result_key, result in results.items():
            st.subheader(f"{result_key}")
            st.write("Classification Report:")
            if isinstance(result, dict):
                for class_name, metrics in result.items():
                    if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                        continue
                    st.write(f"Class: {class_name}")
                    df = pd.DataFrame(metrics, index=[0])
                    st.dataframe(df)

    def run(self):
        st.title("Predicting with Deep learning (DL) models")

        dataset_names = st.multiselect("Select Datasets (more than one option possible)", ["MITBIH", "PTBDB"])
        selected_sampling = st.selectbox("Select the sampling method on which the models were trained",
                                         ["A_Original", "B_SMOTE"])
        model_options = ["Simple_ANN", "Simple_CNN", "Advanced_CNN"]
        selected_models = st.multiselect("Select the DL models (more than one option possible)", options=model_options)
        experiment_options = ["1", "2", "3", "4"]
        selected_experiments = st.multiselect("Select the experiments the models were trained on (more than one option possible)",
                                              options=experiment_options)
        comparison_method = st.radio("Select the comparison method", ["Single Row (Random)", "Complete Dataset"])

        self.display_options = st.multiselect("Select Display Option(s)",
                                               ["Classification Report", "Predicted Label", "True Label", "Confusion Matrix"],
                                               default=self.display_options)

        if st.button("Generate Results"):
            st.session_state["all_results"] = {}  # this is necessary, otherwise the results cumulate and we don´t want that.
            self.all_results = st.session_state['all_results']
            if len(dataset_names) > 1:
                st.warning("Comparing between datasets is selected.")
                st.info("Please note that direct comparison metrics may not be meaningful across different datasets.")
                for dataset_name in dataset_names:
                    st.subheader(f"Results for {dataset_name} Dataset")
                    mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace()
                    if dataset_name == "MITBIH":
                        dataset_to_use = mitbih_test
                        num_classes = 5
                    else:
                        dataset_to_use = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1,
                                                                                                                 random_state=42)
                        num_classes = 2
                    self.generate_results_and_display(dataset_name, dataset_to_use, selected_models,
                                                      selected_experiments,
                                                      comparison_method, selected_sampling, num_classes)
            else:
                for dataset_name in dataset_names:
                    st.subheader(f":green[Results for {dataset_name} Dataset]")
                    mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace()
                    if dataset_name == "MITBIH":
                        dataset_to_use = mitbih_test
                        num_classes = 5
                    else:
                        dataset_to_use = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1,
                                                                                                                 random_state=42)
                        num_classes = 2

                    self.generate_results_and_display(dataset_name, dataset_to_use, selected_models,
                                                      selected_experiments,
                                                      comparison_method, selected_sampling, num_classes)

        if st.button("show all_results (debugging)"):
            st.write("Here comes the debugging outside of if st.button(generate_results)....")
            st.write(self.all_results)


def Page_DL_Stage_2():
    predictor = DLModelPredictor()
    predictor.run()
