import streamlit as st
from functions import load_datasets_in_workingspace, predict_with_DL
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


@st.cache_data()
def generate_results(dataset_name, dataset_to_use, selected_models, selected_experiments,
                     comparison_method, selected_sampling, num_classes):
    results = {}
    single_row_results = {}

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
                single_row_results[f"{model_name}_Exp{experiment}"] = {
                    "y_true": y_true,
                    "prediction": int(prediction[0]),
                    "report": report
                }

            elif comparison_method == "Complete Dataset":
                prediction, report = predict_with_DL(test=dataset_to_use, model=model_name, model_path=model_path,
                                                     show_conf_matr=False, num_classes=num_classes)
                y_true = dataset_to_use[187].values
                results[f"{model_name}_Exp{experiment}"] = report

    return dataset_name, results, single_row_results


@st.cache_data()
def display_classification_report(results):
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


@st.cache_data()
def display_confusion_matrix(results):
    for dataset_name, single_row_class_report_results in results.items():
        st.write("dataset_name:", dataset_name)
        st.write("single_row_class_reports_results:", single_row_class_report_results)
        for model_exp, result in single_row_class_report_results.items():
            y_true = result["y_true"]
            prediction = result["prediction"]
            cm = confusion_matrix(y_true, prediction)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
            ax.set_title(f"Confusion Matrix for {model_exp} and Dataset {dataset_name}")
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            st.pyplot(fig)


def run():
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

    display_options = st.multiselect("Select Display Option(s)",
                                     ["Classification Report", "Confusion Matrix"],
                                     default=[])

    if st.button("Generate Results"):
        st.session_state["all_results"] = {}
        all_results = st.session_state['all_results']
        for dataset_name in dataset_names:
            st.subheader(f"Results for {dataset_name} Dataset")
            mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace()
            dataset_to_use = mitbih_test if dataset_name == "MITBIH" else pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1, random_state=42)
            num_classes = 5 if dataset_name == "MITBIH" else 2
            dataset_name, results, single_row_results = generate_results(dataset_name, dataset_to_use, selected_models,
                                                                          selected_experiments, comparison_method, selected_sampling, num_classes)
            all_results[dataset_name] = single_row_results
            st.session_state['all_results'] = all_results
            st.write("Here comes the debugging for the generated results....")
            st.write(all_results)

        if "Classification Report" in display_options:
            st.write("Displaying Classification Report...")
            display_classification_report(results)

        if "Confusion Matrix" in display_options:
            st.write("Displaying Confusion Matrix with single row results")
            display_confusion_matrix(single_row_results)

    if st.button("Show Results"):
        st.write("Displaying Results...")
        st.write(all_results)

def Page_DL_Stage_2():
    run()
