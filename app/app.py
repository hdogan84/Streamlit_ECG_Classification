# Core Pkg
import streamlit as st


# Custom modules
from Introduction_page import Introduction # Introduction page function
from Data_viz_preprocessing import Data_viz_preprocessing # Basic ML web app with stremlit
from Predict_with_ML_models import Page_ML_Stage_1
#from Predict_with_DL_models import predict_with_DL #can be deleted?
from Predict_with_DL_models import Page_DL_Stage_2

def main():

    # List of pages
    liste_menu = ["Introduction to the Problem", "Data Vizualization and Preprocessing", "Modeling Stage 1", "Modeling Stage 2", "Comparisons", "Conclusion and Outlook", "Team Members"]

    # Sidebar
    menu = st.sidebar.selectbox("Content Selection", liste_menu)
    
    # Page navigation
    if menu == liste_menu[0]:
        Introduction()
    elif menu == liste_menu[1]:
        Data_viz_preprocessing()
    elif menu == liste_menu[2]:
        st.title(" Modeling Stage 1 (ML Models)")
        Page_ML_Stage_1()
    elif menu == liste_menu[3]:
        st.title("Here will be some model selection for Modeling Stage 2 (DL Models)")
<<<<<<< HEAD
        Page_DL_Stage_2()
                
=======
        predict_with_DL()
        st.header("therefore a new submodule with specific functions has to be created.")
        #st.header(":red[Notes for further Improvement:]")
        #st.write("Make a selection routine instead of predefined tables:")
        #st.write("- Select Dataset")
        #st.write("- select the ML Models that should be compared together")
        #st.write("- Select the comparison Method:")
        #st.write("- - Single Row: Use a single row (random) to predict --> Compare the real class with the predicted classes of the selected models.")
        #st.write("- - Complete Dataset: Print classification reports, confusion matrix, bar plots with metrics for each model selected or find a way to show all results in one single plot (like results plot in the report)")
        #st.write("- Print / plot the results")
        predict_with_DL()

        
>>>>>>> 146722fa964fc57aef27a771531117cc593c04f2
    elif menu == liste_menu[4]:
        st.title("Here could be a submodule that allows some comparisons between the models and creates vizualizations on the go (purely optional, since we have absolutely no code for this yet!)")
        st.header("therefore a new submodule with specific functions has to be created.")
    elif menu == liste_menu[5]:
        st.title("Here we present our conclusions and outlooks (this can be just a short summary of the final report anyway and emails for job inquiries.))")
        st.header("therefore a new submodule with specific functions has to be created.")
    elif menu == liste_menu[6]:
        st.title("Here we present the Members of our Team.")
        st.header("therefore a new submodule with specific functions has to be created. It could link to the linked in page etc.")
        st.write("Simon Dommer:")
        st.markdown("Expert in nothing.")

if __name__ == '__main__':
    main()