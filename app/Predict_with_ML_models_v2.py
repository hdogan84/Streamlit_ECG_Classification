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
from functions import load_pkl_model, predict_with_ML 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def Page_ML_Stage_1(data_path = "../data/heartbeat"):
    """
    Function to display the page for the modeling with the ML models.
    - data_path = path where the datasets mitbih and ptbdb are stored (normally ../data/heartbeat)

    """

    ### Create Title
    st.title("Predicting with ML")

    #this should be done in the second page, and also ptbdb data needs to be concated and shuffled --> also shown in page 2
    mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace(data_path)
    
        #Checkbox for the button
    st.subheader("MITBIH Predictions")
    #we now only call the designated prediction function!
    predict_with_ML(mitbih_test)

    st.subheader(":red[PTBDB Predictions (on ptbdb normal)]")
    #we now only call the designated prediction function!
    predict_with_ML(ptbdb_normal)

    st.subheader(":red[PTBDB Predictions (on ptbdb abnormal)]")
    #we now only call the designated prediction function!
    predict_with_ML(ptbdb_abnormal)

    st.header(":red[Notes for further Improvement:]")
    st.write("Make a selection routine instead of predefined tables:")
    st.write("- Select Dataset")
    st.write("- select the ML Models that should be compared together")
    st.write("- Select the comparison Method:")
    st.write("- - Single Row: Use a single row (random) to predict --> Compare the real class with the predicted classes of the selected models.")
    st.write("- - Complete Dataset: Print classification reports, confusion matrix, bar plots with metrics for each model selected or find a way to show all results in one single plot (like results plot in the report)")
    st.write("- Print / plot the results")


def Page_ML_Stage_2(data_path = "../data/heartbeat"):
    
    st.title("Predicting with Machine learning (ML) models")

    #Here some dynamic overviews are needed! with checkboxes to hide them.
    # --> Overview on model structures: Call build function and then model.summary?
    # --> Overview on experiment design: Make a table in functions.py or complete function just for the table and show it? Or use picture at first from the report!

    #choose the dataset --> Multiselection, so comparison of different datasets is also possible
    dataset_names = st.multiselect("Select Datasets (more than one option possible)", ["MITBIH", "PTBDB"])

    gridsearch_options = ['Optimized_Model_with_Gridsearch','Basemodel_no_gridsearch']
    if "PTBDB" in dataset_names: 
        gridsearch_options = ['Basemodel_no_gridsearch']

    gridsearch_select = st.selectbox("Select Gridsearch On/Off", gridsearch_options)

    st.write(gridsearch_select)
    
    sampling_options = ["A_Original", "B_SMOTE", "C_RUS"]
    if gridsearch_select == 'Optimized_Model_with_Gridsearch': 
        sampling_options = ["A_Original"]

    #choose the sampling method (Only original and B_SMOTE should be available)
    selected_sampling = st.selectbox("Select the sampling method on which the models were trained", sampling_options)
    #choose the models --> Multiselection
    model_options = ["SVM", "DTC", "XGB"]
    selected_models = st.multiselect("Select the DL models (more than one option possible)", options=model_options)
    selected_experiments = 1100 #Parameter not in use
    
    #choose the comparison method (only one selection possible)
    comparison_method = st.radio("Select the comparison method", ["Single Row (Random)", "Complete Dataset"])

    mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal = load_datasets_in_workingspace()

    all_results = {} #empty dictionary, that is used in either of the two branches below
    if len(dataset_names) > 1:
        st.warning("Comparing between datasets is selected.")
        st.info("Please note that direct comparison metrics may not be meaningful across different datasets.")
        for dataset_name in dataset_names:
            st.subheader(f"Results for {dataset_name} Dataset")  
            if dataset_name == "MITBIH":
                dataset_to_use = mitbih_test
                num_classes = 5
            else:
                #here the concatenting and reshuffling is done
                dataset_to_use = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1, random_state=42)
                num_classes = 2
            results = generate_results(dataset_name, dataset_to_use, selected_models, gridsearch_select, comparison_method, selected_sampling, num_classes)
            # Store results in a dictionary with dataset_name as key
            all_results[dataset_name] = results

    else:
        for dataset_name in dataset_names:
            st.subheader(f":green[Results for {dataset_name} Dataset]")
            if dataset_name == "MITBIH":
                dataset_to_use = mitbih_test
                num_classes = 5 #we can simply choose the num_classes here isntead of overcomplicating things?
            else:
                #here the concatenting and reshuffling is done
                dataset_to_use = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True).sample(frac=1, random_state=42)
                num_classes = 2
            
            results = generate_results(dataset_name, dataset_to_use, selected_models, gridsearch_select, comparison_method, selected_sampling, num_classes)
            # Store results in a dictionary with dataset_name as key
            all_results[dataset_name] = results            
            
    
#If this works, the function should be outsourced in functions.py
#but first: Make it work entirely and then rename it, so it doesnt get lost in functions.py
@st.cache_data
def generate_results(dataset_name, dataset_to_use, selected_models, gridsearch_select, comparison_method, selected_sampling, num_classes):
    results = {}
    
    for model_name in selected_models:
        
        model_path = f"../data/ML_Models/{model_name}_{gridsearch_select}_{dataset_name}_{selected_sampling}.pkl"

        if comparison_method == "Single Row (Random)":
            row_index = np.random.randint(len(dataset_to_use))
            single_row = pd.DataFrame(dataset_to_use.iloc[row_index].values.reshape(1, -1)) #has to be dataframe for predict_with_dl function, even if its only single row.
            y_true = dataset_to_use.iloc[row_index, 187]
            prediction, report = predict_with_ML(test=single_row, model_file_path=model_path, show_conf_matr=False)
            st.write(f"Model: {model_name}, Exp, True Label: {y_true}, Predicted Label: {prediction}")
            results[f"{model_name}"] = report
            st.write(f"Classification report for {model_name} ")
            st.dataframe(report)

        elif comparison_method == "Complete Dataset":
            prediction, report = predict_with_ML(test=dataset_to_use, model_file_path=model_path, show_conf_matr=False)
            y_true = dataset_to_use[187].values
            results[f"{model_name}"] = report
            st.write(f"Classification report for {model_name} ")
            st.dataframe(report)

    #the plotting of the confusion matrizes is only done for complete dataset, otherwise it makes no sense. But even here, it doesnt provide too much benefit.
    if comparison_method == "Complete Dataset":
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
    return results
         
