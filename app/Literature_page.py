# Core Pkg
import streamlit as st
from functions import displayPDF # our own function to display pdf files

def Literature():
    
    from PIL import Image
    st.header("Comparison with other models on Kaggle")
    img = Image.open("../assets/ComparisonLiterature.png")
    # Plot the image
    st.image(img, caption="Fig 12: Literature review and comparison with our own CNN Models for the heartbeat classification task.")
    
    ## References
    st.header("References")
    
    ref_text = """
1. S. Fazeli, ECG Heartbeat Categorization Dataset,  https://www.kaggle.com/datasets/shayanfazeli/heartbeat. Last accessed on 16.03.2024. 
2. M. Kachuee, S. Fazeli, and M. Sarrafzadeh,  ECG Heartbeat Classification: A Deep Transferable Representation, https://arxiv.org/pdf/1805.00794.pdf. Last accessed on 16.03.2024.
3. PhysioNet PTBDB ANN, https://github.com/anandprems/physionet_ptbdb_ann. Last accessed on 15.03.2024. 
4. Mahfuj Hossain, Kaggle user. XGB classification algorithm.  https://www.kaggle.com/code/mahfujhossain/ecg-xgb-unbalanced 
5. Mahfuj Hossain, Kaggle user. DNN classification algorithm.  https://www.kaggle.com/code/mahfujhossain/ecg-imbalanced-dataset-mlp 
6. Kanhaiyachatla, Kaggle user. CNN - LSTM classification algorithm.  https://www.kaggle.com/code/kanhaiyachatla/ecg-classification-using-cnn-lstm 
7. Erfan Saeedi, Kaggle user. CNN and Transformer classification algorithms.  https://www.kaggle.com/code/erfansaeedi/cnn-and-transformer#CNN  
8. Gregoire DC, Kaggle user. CNN algorithm.  https://www.kaggle.com/code/gregoiredc/arrhythmia-on-ecg-classification-using-cnn   
9. Zakari, Kaggle user. CNN Autoencoder algorithm. https://www.kaggle.com/code/redpen12/anomaly-detection-with-cnn-autoencoders#Building-CNN-Autoencoder-Model   
10. Nicolas MINE, Kaggle user. CNN algorithm with data augmentation. https://www.kaggle.com/code/coni57/model-from-arxiv-1805-00794   
11. Marco Polo, Kaggle user. CNN - LSTM - Attention algorithm. https://www.kaggle.com/code/polomarco/ecg-classification-cnn-lstm-attention-mechanism   


"""
    
    st.text(ref_text)
    
    