# Core Pkg
import streamlit as st
from functions import load_datasets_in_workingspace, predict_with_DL
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def Page_DL_Stage_2(data_path = "../data/heartbeat"):
    
    st.title("Predicting with Deep learning (DL) models")

    #Here some dynamic overviews are needed! with checkboxes to hide them.
    # --> Overview on model structures: Call build function and then model.summary?
    # --> Overview on experiment design: Make a table in functions.py or complete function just for the table and show it? Or use picture at first from the report!

    #choose the dataset --> Multiselection, so comparison of different datasets is also possible
    dataset_names = st.multiselect("Select Datasets (more than one option possible)", ["MITBIH", "PTBDB"])

    #choose the sampling method (Only original and B_SMOTE should be available)
    selected_sampling = st.selectbox("Select the sampling method on which the models were trained", ["A_Original", "B_SMOTE"])
    #choose the models --> Multiselection
    model_options = ["Simple_ANN", "Simple_CNN", "Advanced_CNN"]
    selected_models = st.multiselect("Select the DL models (more than one option possible)", options=model_options)

    #choose the experiment that was used to train the models, also multiselection to make it fun.
    #this needs an (dynamic!) overview figure of the experiments, so that the user has knowledge what this means
    experiment_options = ["1", "2", "3", "4"] #does this make sense with strings or are integers preferable?
    selected_experiments = st.multiselect("Select the experiments the models were trained on (more than one option possible)", options=experiment_options)

    #choose the comparison method (only one selection possible)
    comparison_method = st.radio("Select the comparison method", ["Single Row (Random)", "Complete Dataset"])


    if len(dataset_names) > 1:
        st.warning("Comparing between datasets is selected.")
        st.info("Please note that direct comparison metrics may not be meaningful across different datasets.")
        for dataset_name in dataset_names:
            st.subheader(f"Results for {dataset_name} Dataset")
            mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace()
            if dataset_name == "MITBIH":
                dataset_to_use = mitbih_test
                st.dataframe(dataset_to_use) #for debugging
            else:
                #here the concatenting and reshuffling is done
                dataset_to_use = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1, random_state=42)
                st.dataframe(dataset_to_use) #for debugging
            # Code hier einfügen, kann von weiter unten kopiert werden und geringfügig angepasst werden.
            # Dann am Ende eine Funktion aus dem ganzen Mist hier schreiben und weg damit?

    else:
        for dataset_name in dataset_names:
            st.subheader(f":green[Results for {dataset_name} Dataset]")
            mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace()
            if dataset_name == "MITBIH":
                dataset_to_use = mitbih_test
                num_classes = 5 #we can simply choose the num_classes here isntead of overcomplicating things?
                #st.dataframe(dataset_to_use) #for debugging
            else:
                #here the concatenting and reshuffling is done
                dataset_to_use = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1, random_state=42)
                num_classes = 2
                #st.dataframe(dataset_to_use) #for debugging
                            
            y_true = dataset_to_use[187].values
            
            if comparison_method == "Single Row (Random)":
                row_index = np.random.randint(len(dataset_to_use))
                single_row = pd.DataFrame(dataset_to_use.iloc[row_index].values.reshape(1, -1)) #has to be dataframe for predict_with_dl function, even if its only single row.
                true_label = y_true[row_index]
                
                for model_name in selected_models:
                    for experiment in selected_experiments:
                        #this is the model path that should be reused in that way!
                        model_path = f"../assets/DL_Models/{model_name}/experiment_{experiment}_{dataset_name}_{selected_sampling}.weights.h5"
                        prediction, report = predict_with_DL(test=single_row, model=model_name, model_path=model_path, show_conf_matr=False, num_classes=num_classes) #conf matrix makes no sense for one prediction
                        st.write(f"Model: {model_name}, Experiment: {experiment}, True Label: {true_label}, Predicted Label: {prediction}")
                        st.write("Classification report")
                        st.dataframe(report)

            elif comparison_method == "Complete Dataset":
                results = {}
                for model_name in selected_models:
                    for experiment in selected_experiments:
                        #this is the model path that should be reused in that way!
                        model_path = f"../assets/DL_Models/{model_name}/experiment_{experiment}_{dataset_name}_{selected_sampling}.weights.h5"
                        prediction, report = predict_with_DL(test=dataset_to_use, model=model_name, model_path=model_path, show_conf_matr=False, num_classes=num_classes) #conf matrix is done below 
                        st.write(f"Model: {model_name}, Experiment: {experiment}") #, True Label: {true_label}, Predicted Label: {prediction}
                        st.write("Classification report")
                        st.dataframe(report)
                        results[f"{model_name}_Exp{experiment}"] = classification_report(y_true, prediction, output_dict=True)

                for result_key, result in results.items():
                    #st.subheader(f"{result_key}")
                    #st.json(result)

                    # Confusion matrix, not done with dedicated function, but beautiful?
                    cm = confusion_matrix(y_true, prediction)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
                    ax.set_title(f"Confusion Matrix for {result_key} and Dataset {dataset_name}_{selected_sampling}")
                    ax.set_xlabel('Predicted Labels')
                    ax.set_ylabel('True Labels')
                    st.pyplot(fig)
    

         
