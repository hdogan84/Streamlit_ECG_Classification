# Core Pkg
import streamlit as st
from functions import displayPDF # our own function to display pdf files

def Outlook():
    st.header("Conclusions")
    st.write("- ML and DL Models both performed very well in comparison with recent literature.")
    st.write("- ML and DL Models both performed better on the PTBDB Dataset. This could be due to the smaller amount of classes to be distinquished.")
    st.write("- :red[PLEASE FILL THIS WITH CONCLUSIONS FROM THE REPORT]")


    st.header("Outlook")
    st.write("The best DL Model in experiment 4 configuration can be deployed for the following application areas.")
    st.write("- Assistance in classifying heartbeat patterns")
    st.write("- long term observation of patients or during night shifts")
    st.write("- Training of prospective medical staff via a website / app")
    st.write("- :red[PLEASE FILL THIS WITH OUTLOOKS FROM THE REPORT OR YOUR OWN MIND AFTER PLAYING AROUND WITH THE DEMO]")
    
    