import streamlit as st
import base64
import pandas as pd 
import seaborn as sns 
import streamlit as st 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pickle
import inspect
import plotly.graph_objects as go
import plotly.express as px
from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow as tf #Version 2.13.0 is required since this was used by Kaggle to produce the .weights.h5 files
#PUT THIS INTO REQUIREMENTS.TXT --> Tensorflow MUST be 2.13.0!!! We don´t really need to import tensorflow, but it must be installed as version 2.13.0
#import tensorflow.keras as keras #can be deleted for ubuntu?
import xgboost

st.set_option('deprecation.showPyplotGlobalUse', False) #removing errors from beeing shown.
#known futurewarnings: st.pyplot() should not be called without arguments (e.g. "fig").

"""
Important Notes:
- the filepaths have to be relative, i.e. like this "../assets/[file in the assets folder]". In order to use this correctly, the command "streamlit runn app.py" has to be used
--> Therefore one has to navigate directly into the folder where the app.py folder is located! --> This is good to remember for now, but especially good to remember when trying to deploy our app?

"""

def displayPDF(file, width=1000, height=700, caption=None):
    """
    Takes a pdf file as argument and shows it as st.markdown directly.
    The Argument <<file>> must therefore be also a direct path to the file.
    """
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" type="application/pdf"></iframe>'
    
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

    # Displaying Caption
    if caption:
        st.write(f'<p style="font-size: small; color: #888888;">{caption}</p>', unsafe_allow_html=True)

    
# Custom function to train models --> Can be removed later, but serves as an example for now
# st.cache is used to load the function into memory, i.e. it will be checked, if the function has been called with the specific parameters already.
@st.cache_data
def train_model(model_choisi, X_train, y_train, X_test, y_test) :
    if model_choisi == 'Regression Logisitic' : 
        model = LogisticRegression()
    elif model_choisi == 'Decision Tree' : 
        model = DecisionTreeClassifier()
    elif model_choisi == 'KNN' : 
        model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score

@st.cache_data
def load_pkl_model(model_path_pkl):
    with open(model_path_pkl, "rb") as pickle_file:
        model = pickle.load(pickle_file)
    return model

@st.cache_data
def download_datasets(download_path, dataset_owner="shayanfazeli", dataset_name="heartbeat"):
    """
    Downloads datasets from Kaggle API and stores them in the specified folder.

    Args:
    - dataset_owner (str): Owner of the dataset on Kaggle.
    - dataset_name (str): Name of the dataset on Kaggle.
    - download_path (str): Path to the download folder outside the Git repository.
    """
    # Configure and authenticate with the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Check if the dataset folder already exists
    if not os.path.exists(download_path):
        # Dataset folder does not exist --> Download and save the datasets
        api.dataset_download_files(dataset_owner + "/" + dataset_name, path=download_path, unzip=True) #=os.path.join(download_path, dataset_name)
        st.write("Datasets are downloaded and unzipped.")
    else:
        # Dataset folder exists, but datasets might be missing
        missing_files = [] 
        for file_name in ["mitbih_test.csv", "mitbih_train.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"]:  
            file_path = os.path.join(download_path, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)

        if missing_files:
            # If missing files are present, download ALL files and overwrite the old folder.
            api.dataset_download_files(dataset_owner + "/" + dataset_name, path=download_path, unzip=True, force=True)# =os.path.join(download_path, dataset_name)
            st.write("Missing data was downloaded and unzipped. All Datasets are now available.")
        else:
            st.write("All Datasets are already available.")

@st.cache_data
def show_download_code_Kaggle():
    with st.expander("Show download_datasets function code"):
            st.code(
                """
    import os
    from kaggle.api.kaggle_api_extended import KaggleApi

    def download_datasets(download_path, dataset_owner="shayanfazeli", dataset_name="heartbeat"):
        # Configure and authenticate with the Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Check if the dataset folder already exists
        dataset_folder = os.path.join(download_path, dataset_name)
        if not os.path.exists(dataset_folder):
            # Dataset folder does not exist --> Download and save the datasets
            api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True)
            print("Datasets are downloaded and unzipped.")
        else:
            # Dataset folder exists, but datasets might be missing
            missing_files = [] 
            for file_name in ["mitbih_test.csv", "mitbih_train.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"]:  
                file_path = os.path.join(dataset_folder, file_name)
                if not os.path.exists(file_path):
                    missing_files.append(file_name)

            if missing_files:
                # If missing files are present, download ALL files and overwrite the old folder.
                api.dataset_download_files(dataset_owner + "/" + dataset_name, path=dataset_folder, unzip=True, force=True)
                print("Missing data was downloaded and unzipped. All Datasets are now available.")
            else:
                print("All Datasets are already available.")
                """
            )
            st.caption("Code 1: Code used to download the Datasets from Kaggle. Authentification might be required.")
    

@st.cache_data #The path_to_datasets might be specified in a dynamic .env file?
def load_datasets_in_workingspace(path_to_datasets="../data/heartbeat"):
    #reading in the datasets from the local ../data folder --> this folder is not pushed on github and only locally available.
    mitbih_test = pd.read_csv(path_to_datasets + "/" + "mitbih_test.csv",header=None)
    mitbih_train = pd.read_csv(path_to_datasets + "/" + "mitbih_train.csv",header=None)
    ptbdb_abnormal = pd.read_csv(path_to_datasets + "/" + "ptbdb_abnormal.csv",header=None)
    ptbdb_normal = pd.read_csv(path_to_datasets + "/" + "ptbdb_normal.csv",header=None)
    return mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal


@st.cache_data
def calculate_average_values(dataset_folder, dataset_names):
    """
    Calculates average values for each class in each dataset.
    Is used in the plot_random_row function --> The calculation of the mean values is always the same and must not be repeated!
    Args:
    - dataset_folder (str): Path to the folder containing the datasets.
    - dataset_names (list of str): List of dataset names.

    Returns:
    - average_values (dict): Dictionary containing average values for each class in each dataset.
    """
    average_values = {}
    for dataset_name in dataset_names:
        df = pd.read_csv(f"{dataset_folder}/{dataset_name}")
        df_columns = df.columns.tolist()
        target_column = df_columns[-1]
        average_values[dataset_name] = df.groupby(target_column).mean().values
    return average_values

#@st.cache_data is not necessary, because this function is always called with random row numbers?
def plot_random_row(dataset_folder = "/home/simon/Datascientest_Heartbeat/jan24_bds_int_heartbeat/data/KAGGLE_datasets/heartbeat", dataset_names = ["mitbih_test.csv", "mitbih_train.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"]):
    """
    Plots a random row from one of the datasets.

    Args:
    - dataset_folder (str): Path to the folder containing the datasets.
    - dataset_names (list of str): List of dataset names.
    """
    # Load average values
    average_values = calculate_average_values(dataset_folder, dataset_names)
    #dictionary for classes --> This should be a globally available dict, so that it cannot be changed from somewhere else.
    disease_names_mitbih = {0: "Normal", 1: "Supraventricular", 2: "Ventricular", 3: "Fusion V and N", 4: "Paced / Fusion Paced and Normal / Unknown"}
    disease_names_ptbdb = {0: "Normal", 1: "Abnormal"}
    # Select a random dataset
    selected_dataset = np.random.choice(dataset_names)

    # Load the dataset
    dataset_path = f"{dataset_folder}/{selected_dataset}"
    df = pd.read_csv(dataset_path)

    # Define custom color palette for dashed lines
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Select a random row
    random_row_index = np.random.randint(len(df))
    random_row = df.iloc[random_row_index]

    # Plot the random row
    plt.figure(figsize=(10, 6))
    plt.plot(random_row[:-1], label="Random Row")  # Exclude the last column (target)
     # Plot average values for each class in the background with dashed lines
    if "ptbdb" in selected_dataset:
        ptbdb_datasets = {"ptbdb_normal.csv": "Normal", "ptbdb_abnormal.csv": "Abnormal"}
        for i, (dataset, class_label) in enumerate(ptbdb_datasets.items()):
            class_avg = average_values[dataset][0] #this is the key to make it work --> THe output list was doubled [[]], i don´t exactly know why
            plt.plot(class_avg, linestyle='--', color=color_palette[i], alpha=0.8, label=f"{class_label} Average")
    else:
        for i, class_avg in enumerate(average_values[selected_dataset]):
            class_label = disease_names_mitbih[i]
            plt.plot(class_avg, linestyle='--', color=color_palette[i], alpha=0.8, label=f"{class_label} Average")

    plt.title(f"Random Row ({random_row_index}) from {selected_dataset} Dataset")
    plt.xlabel("Timesteps in ms")
    plt.ylabel("Normalized ECG Amplitude")
    plt.grid(True)
    x_ticks_positions = np.arange(0, 188, 10) #Positions of the x-ticks (we choose an tick for every 10th position)
    x_tick_labels = [i * 8 for i in x_ticks_positions] #each timestep is 8ms long, already multiplied by 10 due to x_ticks_positions..
    plt.xticks(x_ticks_positions, x_tick_labels, rotation=90) #we can now change the x_ticks accordingly.
    plt.legend()
    st.pyplot()

    # Show the target column with disease names
    dataset_disease_names = disease_names_mitbih if selected_dataset == "MITBIH" else disease_names_ptbdb
    target_value = random_row.iloc[-1]
    heartbeat_class = dataset_disease_names.get(target_value, "Unknown")
    st.caption("Fig 5: Random Plot of a heartbeat shape from one of the datasets. Averages of residual classes shown.")
    st.text(f"Classification of Heartbeat: {heartbeat_class}")


@st.cache_data
def predict_with_ML(test, model_file_path="../assets/RFC_Optimized_Model_with_Gridsearch_MITBIH_A_Original.pkl", show_conf_matr=True, cm_title="Confusion Matrix", xtick_labels=None, ytick_labels=None):
    """
    Function to be used for the prediction with the simple ML models
    Assumption: The needed files are already available and correctly named. Otherwise further Arguments need to be introduced.
    - test: The dataset (test) that is used for the predictions.
    - model_file_path: The path to the .pkl file that is used for predictions
    - show_conf_matr == True: switch to show confusion matrix or to not show it.
    - cm_title: Title for the confusion matrix plot
    - xtick_labels: List of labels for x-axis ticks
    - ytick_labels: List of labels for y-axis ticks
    """

    y_test = test[187]
    X_test = test.drop(187, axis=1)

    #model = load_pkl_model(model_file_path)

    with open(model_file_path, "rb") as pickle_file:
        model = pickle.load(pickle_file)
    

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, digits=4, output_dict=True)
    #st.dataframe(report)  # showing the classification report as dataframe --> Beautiful.

    if show_conf_matr:
        #this space is needed, no one knows why...
        show_conf_matrix(y_test, predictions, cm_title=cm_title, xtick_labels=xtick_labels, ytick_labels=ytick_labels)

    return predictions, report

@st.cache_data
def show_conf_matrix(y_true, y_pred, cm_title="Confusion Matrix", xtick_labels=None, ytick_labels=None):
    """
    Function to show confusion matrix.
    - y_true: True labels.
    - y_pred: Predicted labels.
    - cm_title: Title for the confusion matrix plot.
    - xtick_labels: List of labels for x-axis ticks (optional).
    - ytick_labels: List of labels for y-axis ticks (optional).
    """

    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Generate xtick labels if not provided
    if xtick_labels is None:
        unique_labels = sorted(set(y_true).union(set(y_pred)))
        xtick_labels = [f"Class {label}" for label in unique_labels]
                
    # Generate ytick labels if not provided
    if ytick_labels is None:
        ytick_labels = xtick_labels
    fig, ax = plt.subplots()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=xtick_labels, yticklabels=ytick_labels, ax=ax)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(cm_title) #title is not working correctly?
    st.pyplot(fig)

@st.cache_data
def predict_with_DL(test, model="Advanced_CNN",  model_path = "/home/simon/demo_streamlit_jan22cds_en/assets/experiment_4_MITBIH_A_Original.weights.h5", num_classes = 5, show_conf_matr=True, cm_title="Confusion Matrix", xtick_labels=None, ytick_labels=None):
    """
    Function to be used for the prediction with the DL models
    Assumption: The needed files are already available and correctly named. Otherwise further Arguments need to be introduced.
    - test: The dataset (test) that is used for the predictions.
    - model_path: The path to the .h5 file that is used for predictions
    - show_conf_matr == True: switch to show confusion matrix or to not show it.
    - cm_title: Title for the confusion matrix plot
    - xtick_labels: List of labels for x-axis ticks
    - ytick_labels: List of labels for y-axis ticks
    """
    y_test = test[187]
    X_test = test.drop(187,axis=1)
    num_classes = num_classes #we get this from the script that calls this function conveniently
    
    #checking which model was selected (simple_ann, simple_cnn or advanced_cnn)
    if model == "Advanced_CNN":
        model = build_model_adv_cnn(model_path, num_classes=num_classes)
        predictions = model.predict(X_test).argmax(axis=1)
        report = classification_report(y_test, predictions, digits=4, output_dict=True)
        #st.dataframe(report) #this can be muted if the report is returned from the function
    elif model == "Simple_CNN":
        model = build_model_simple_cnn(model_path, num_classes=num_classes)
        predictions = model.predict(X_test).argmax(axis=1)
        report = classification_report(y_test, predictions, digits=4, output_dict=True)
        #st.dataframe(report) #this can be muted if the report is returned from the function
    elif model == "Simple_ANN":
        model = build_model_simple_ann(model_path, num_classes=num_classes)
        predictions = model.predict(X_test).argmax(axis=1)
        report = classification_report(y_test, predictions, digits=4, output_dict=True)
        #st.dataframe(report) #this can be muted if the report is returned from the function
    else:
        st.write("Debugg Message: Model selection in the code was not successful!")
    if show_conf_matr:
        #this space is needed, no one knows why...
        show_conf_matrix(y_test, predictions, cm_title=cm_title, xtick_labels=xtick_labels, ytick_labels=ytick_labels)
    return predictions, report

# build advanced CNN model and load weights from h5 file (Deep learning)
@st.cache_data
def build_model_adv_cnn(model_path, num_classes=5):
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
    
    adv_cnn_model.load_weights(model_path)
    return adv_cnn_model

@st.cache_data
def build_model_simple_cnn(model_path,num_classes=5):
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
    
    cnn_model.load_weights(model_path)
    return cnn_model

@st.cache_data
def build_model_simple_ann(model_path,num_classes=5):
    """
    builds the simple ANN model from the reports
    """

    ann_model = tf.keras.models.Sequential()
    ann_model.add(tf.keras.layers.Dense(60, activation=tf.keras.layers.LeakyReLU(alpha=0.001), input_shape=(187,)))
    ann_model.add(tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    ann_model.add(tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    ann_model.add(tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    ann_model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) #softmax classes are dynamically adjusted according to the dataset!
    
    ann_model.load_weights(model_path)
    return ann_model


##### For now only used for Predict_with_DL_models
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

##### For now only used for Predict_with_DL_models
@st.cache_data
def  display_classification_report(results):
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

##### For now only used for Predict_with_DL_models
@st.cache_data
def display_confusion_matrix(results):
    for dataset_name, dataset_results in results.items():
        st.subheader(f"Dataset {dataset_name}")
        for model_name, model_result in dataset_results.items():
            st.subheader(f"Model {model_name}")
            for sampling_method, sampling_method_result in model_result.items():
                st.subheader(f"Sampling Method training Set: {sampling_method}")
                for experiment, experiment_result in sampling_method_result.items():
                    st.subheader(f"{experiment}")
                    st.write("Confusion Matrix:")
                    
                    y_true = experiment_result["y_true"]
                    prediction = experiment_result["prediction"]
                    
                    # Check if y_true is a single number
                    if isinstance(y_true, (int, float)):
                        # Create a color-coded plot for the single prediction
                        fig, ax = plt.subplots(figsize=(6, 2))
                        color = 'green' if prediction == y_true else 'red'
                        ax.text(0.5, 0.5, f'Predicted: {prediction}\nTrue: {y_true}', ha='center', va='center', fontsize=20, color=color)
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        # Generate the confusion matrix plot
                        cm = confusion_matrix(y_true, prediction)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
                        ax.set_title(f"Confusion Matrix for {model_name} in {experiment} on {dataset_name} ({sampling_method})")
                        ax.set_xlabel('Predicted Labels')
                        ax.set_ylabel('True Labels')
                        st.pyplot(fig)

#new display_lineplot function that is using the plotly go object.
def display_lineplot(results):
    data = []  # Eine Liste, um Dictionaries der Daten zu sammeln
    for dataset_name, dataset_results in results.items():
        for model_name, model_results in dataset_results.items():
            for sampling_method, sampling_method_results in model_results.items():
                for experiment, experiment_result in sampling_method_results.items():
                    accuracy = experiment_result["report"]["accuracy"]
                    macro_f1 = experiment_result["report"]["weighted avg"]["f1-score"]
                    precision = experiment_result["report"]["weighted avg"]["precision"]
                    recall = experiment_result["report"]["weighted avg"]["recall"]
                    data.append({
                        'Model': model_name + "_" + experiment,
                        'Dataset': dataset_name + "_" + sampling_method,
                        'Accuracy / Recall': accuracy,
                        'F1': macro_f1, 
                        'Precision': precision, 
                    })

    # Erstelle den DataFrame außerhalb der Schleife
    df_metrics = pd.DataFrame(data)

    # Debugging-Ausgabe
    #st.write("Making of dataframe is finished")
    #st.dataframe(df_metrics)
    
    #new version of lineplotting with plotly
    plot_lineplot(df_metrics)

# Line plotting function from Notebook 1, reused here...
def plot_lineplot(df_metrics):
    # Melt DataFrame
    melted_df = pd.melt(df_metrics, id_vars=['Model', 'Dataset'], var_name='Metric', value_name='Value')
    
    fig = px.line(
        melted_df,
        x='Model',
        y='Value',
        color='Metric',
        symbol= "Dataset",
        line_dash= "Dataset", 
        #facet_row='Dataset',
        #facet_col='Metric',  # Änderung hier
        line_shape='linear',  # Verbindungslinien zwischen den Punkten
        render_mode='svg',    # Verbessert die Darstellung in Streamlit
    )
    
    fig.update_layout(
        title="Weighted Average Metrics for the DL Models across the selected combinations",
        xaxis_title="Model",
        yaxis_title="Metrics",
        legend_title="Display Options",
        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="right", x=1),
    )

    st.plotly_chart(fig)


#new radar charts plot figure (testing and debugging)
# for now this function only works withoug class distinction but with the weighted averages
# but: This function would be very good for class wise comparisons!
#@st.cache #not useful in this case
def display_radar_charts(results):
    data = []  # Eine Liste, um Dictionaries der Daten zu sammeln
    for dataset_name, dataset_results in results.items():
        for model_name, model_results in dataset_results.items():
            for sampling_method, sampling_method_results in model_results.items():
                for experiment, experiment_result in sampling_method_results.items():
                    accuracy = experiment_result["report"]["accuracy"]
                    macro_f1 = experiment_result["report"]["weighted avg"]["f1-score"]
                    precision = experiment_result["report"]["weighted avg"]["precision"]
                    recall = experiment_result["report"]["weighted avg"]["recall"]
                    data.append({
                        'Dataset': dataset_name + "_" + sampling_method, #otherwise no distinction is possible for now
                        'Model': model_name + "_" + experiment, #otherwise no distinction is possible for now
                        #'Sampling Method': sampling_method, # this confuses the old function from notebook 1
                        'Accuracy': accuracy,
                        'F1': macro_f1, 
                        'Precision': precision, 
                        'Recall': recall,
                    })

    # Erstelle den DataFrame außerhalb der Schleife
    df_metrics = pd.DataFrame(data)

    # now call the modified function from notebook 1 to display the line plot... --> Rename function or make bar plot out of it.
    plot_radar_chart(df_metrics, 'Model', ['Accuracy', 'F1', 'Precision', 'Recall'])

# Radar plotting function
def plot_radar_chart(data, category_col, metrics):
    """
    Plot a radar chart with customizable parameters.

    Parameters:
    - data: DataFrame, the data to plot.
    - category_col: str, the column name for the categories.
    - metrics: list, a list of column names for the metrics to plot on the radar chart.
    """
    # Create a list of colors for each metric
    colors = sns.color_palette('deep', len(metrics))

    # Create a radar chart for each category in the dataset
    for category in data[category_col].unique():
        category_data = data[data[category_col] == category]
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable and add labels
        plt.xticks(angles, metrics)

        # Draw ylabels
        ax.set_rscale('log')
        ax.set_yticklabels([])

        # Plot each individual radar chart
        for i, metric in enumerate(metrics):
            values = category_data[metric].values.tolist()
            values += values[:1]
            angles_adjusted = angles + angles[:1]  # Adjust angles to match the length of values
            ax.plot(angles_adjusted, values, color=colors[i], linewidth=2, linestyle='solid', label=metric)
            ax.fill(angles_adjusted, values, color=colors[i], alpha=0.4)

        # Add a title
        ax.set_title(category, size=20, color='black', y=1.1)

        # Add a legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        st.pyplot(fig)

@st.cache_data
def plot_with_outliers(data, lower_quantile, upper_quantile):
    """"
    Plots the outliers data for the visualization page
    """
    q1 = data.quantile(lower_quantile)
    q3 = data.quantile(upper_quantile)
    iqr = q3 - q1
    
    # Daten ohne Ausreißer
    not_outliers = data[~((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr)))]
    
    # Plot
    num_classes = data[187].drop_duplicates()
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, class_item in enumerate(num_classes):
        class_data = data[data[187] == class_item]
        plt.plot(class_data.mean(axis=0)[:187], color=color_palette[i], linestyle='--',  label=f"Class {class_item} with outliers")
        plt.plot(not_outliers[not_outliers[187] == class_item].mean(axis=0)[:187], color=color_palette[i], linestyle='-', label=f"Class {class_item} without outliers")

    plt.title("Comparing Mean of Outlier Values")
    plt.xlabel("Timesteps in ms")
    plt.ylabel("Normalized ECG Amplitude")
    plt.grid(True)
    x_ticks_positions = np.arange(0, 188, 10)
    x_tick_labels = [i * 8 for i in x_ticks_positions]
    plt.xticks(x_ticks_positions, x_tick_labels, rotation=90)
    plt.legend(loc='upper right', fontsize='small')  # Setzen Sie die Position und die Größe der Legende
    st.pyplot()
    st.caption("Fig 9: Influence of Outliers on heartbeat shape for variable configurations.")


@st.cache_data
def plot_correlation_matrix(data, selected_dataset):
    """
    plots the correlation matrix just like in the report
    """
    correlation_matrix = data.corr()
    plt.figure(figsize=(16, 16));
    sns.heatmap(correlation_matrix, annot=False, cmap='BrBG', fmt=".2f");
    plt.title(f'Correlation Heatmap of {selected_dataset}');
    st.pyplot()
    st.caption("Fig 10: Correlation Matrix for the selected Dataset.")