# Core Pkg
import streamlit as st
from functions import displayPDF # our own function to display pdf files

def Outlook():
    st.header("Conclusions")
    st.write("- ML and DL models, particularly ANN and CNN, demonstrated strong performance, outperforming recent benchmarks in the literature.")
    st.write("- The best performing model, CNN, achieved an accuracy of 98.57% on the MITBIH dataset and 99.59% on the PTBDB dataset.")
    st.write("- The reliability and legal acceptance of using ML for critical medical applications remain in question. Establishing performance thresholds could be crucial for broader acceptance and to mitigate risks.")
    
    st.header("Outlook")
    st.write("The best DL model from experiment 4 shows potential for deployment in various medical applications:")
    st.write("- Assistance in classifying heartbeat patterns, useful for long-term patient observation or during night shifts.")
    st.write("- Training of prospective medical staff through an interactive website or app.")
    st.write("- To further enhance model performance and applicability, we recommend scaling the model using diverse datasets, adding more convolutional layers, and experimenting with novel data augmentation techniques.")
    st.write("- Special attention should be given to improving precision for detecting abnormal heartbeat patterns, particularly by increasing sample sizes for underrepresented classes.")
    
    