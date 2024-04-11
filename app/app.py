# Core Pkg
import streamlit as st


# Custom modules
from Introduction_page import Introduction # Introduction page function
from Data_viz_preprocessing import Data_viz_preprocessing # Basic ML web app with stremlit
from Predict_with_ML_models import Page_ML_Stage_1
from Predict_with_ML_models_v2 import Page_ML_Stage_2
from DL_Model_Summaries import Page_DL_Stage_1
from Predict_with_DL_models import Page_DL_Stage_2
from Literature_page import Literature # Literatur page function

def main():

    # List of pages
    liste_menu = ["Introduction to the Problem", "Data Vizualization and Preprocessing", 
                 "Machine Learning", "Deep Learning - Models", "Deep Learning - Predictions", 
                 "Comparison with Literature",
                 "Conclusion and Outlook", "Team Members"] #"Comparisons", 

    # Sidebar
    menu = st.sidebar.selectbox("Content Selection", liste_menu)
    
    # Page navigation
    if menu == liste_menu[0]:
        Introduction()
    elif menu == liste_menu[1]:
        Data_viz_preprocessing()
    elif menu == liste_menu[2]:
        st.title(" Modeling Stage 1 (ML Models)")
        #Page_ML_Stage_1()
        Page_ML_Stage_2()
    elif menu == liste_menu[3]:
        st.title("Deep Learning")
        Page_DL_Stage_1()    
    elif menu == liste_menu[4]:
        st.title("Deep Learning")
        Page_DL_Stage_2()                
    elif menu == liste_menu[5]:
        st.title("Comparisons & Bibliography")
        Literature()
    elif menu == liste_menu[6]:
        st.title("Here we present our conclusions and outlooks (this can be just a short summary of the final report anyway and emails for job inquiries.))")
        st.header("therefore a new submodule with specific functions has to be created.")
    elif menu == liste_menu[7]:
        st.title("Here we present the Members of our Team.")
        st.header("therefore a new submodule with specific functions has to be created. It could link to the linked in page etc.")
        st.write("Simon Dommer:")
        st.markdown("Expert in nothing.")

if __name__ == '__main__':
    main()