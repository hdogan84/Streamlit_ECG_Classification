# Core Pkg
import streamlit as st
from functions import displayPDF # our own function to display pdf files

def Introduction():
    st.title("Heartbeat Classification")

    st.write("Our project focuses on analyzing and predicting ECG heartbeat patterns and abnormalities. We've explored various methods, including simple Machine Learning Models and neural networks, using the Kaggle ECG Heartbeat Categorization Dataset.")

    st.header("Key Points:")
    st.markdown("- **Approaches:** Implemented both simple models and neural networks.")
    st.markdown("- **Dataset:** Utilized the Kaggle ECG Heartbeat Categorization Dataset.")
    st.markdown("- **Accuracy:** Achieved high accuracy scores with simple models in predicting overall heartbeat pattern shape and distinguishing between normal and abnormal patterns.")
    st.markdown("- **Neural Networks:** Marginally higher accuracy scores observed, with a tendency towards 'conservativeness' – favoring false positives, which is desirable in medical applications.")
    st.markdown("- **Next Steps:** These initial findings pave the way for further exploration into neural networks' behavior in detecting ECG heartbeat patterns.")

    ## MEDIA
    st.header("Overview on the workflow")
    displayPDF("../assets/Workflow_Heartbeat.pdf", width=800, height=500, caption="The workflow used in the project. Steps with green checks were performed already by the original authors of the main paper.")

    from PIL import Image
    st.header("Dataset Information")
     # open an image
    img = Image.open("../assets/Report1_Table1.png")
    # Plot the image
    st.image(img)

    # MITBIH data
    st.subheader("Classwise Distribution for MITBIH")
    mitbih_text = "MITBIH dataset consists of 5 classes \nwhere the label 'n' shows the 'normal' class"
    st.text(mitbih_text)
    img2 = Image.open("../assets/Report1_Fig1.png")
    # Plot the image
    st.image(img2)
    
    # MITBIH data
    st.subheader("Classwise Distribution for PTBDB")
    ptb_text = """PTBDB dataset consists of normal and abnormal classes\nWe generate a specific test set using Sci-kit Train-Test Split function"""
    st.text(ptb_text)
    img3 = Image.open("../assets/Report1_Fig12_Ptbdb_original.png")
    # Plot the image
    st.image(img3)

    #More Information
    st.header("For More Information on the projects background:")
    st.write("Explore the full original research paper at: [IEEE Xplore](https://ieeexplore.ieee.org/document/8419425)")
    st.write("Explore the original dataset at: [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)")

    st.header(":red[Notes for further Improvement]")
    st.write("Use higher quality pictures or generate the pictures from data. Also include classwise distribution plot for PTBDB.")

