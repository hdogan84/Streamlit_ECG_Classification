# Core Pkg
import streamlit as st

def bases_streamlit():
    st.title("Heartbeat Classification")

    st.write("Our project focuses on analyzing and predicting ECG heartbeat patterns and abnormalities. We've explored various methods, including simple Machine Learning Models and neural networks, using the Kaggle ECG Heartbeat Categorization Dataset.")

    st.header("Key Points:")
    st.markdown("- **Approaches:** Implemented both simple models and neural networks.")
    st.markdown("- **Dataset:** Utilized the Kaggle ECG Heartbeat Categorization Dataset.")
    st.markdown("- **Accuracy:** Achieved high accuracy scores with simple models in predicting overall heartbeat pattern shape and distinguishing between normal and abnormal patterns.")
    st.markdown("- **Neural Networks:** Marginally higher accuracy scores observed, with a tendency towards 'conservativeness' – favoring false positives, which is desirable in medical applications.")
    st.markdown("- **Next Steps:** These initial findings pave the way for further exploration into neural networks' behavior in detecting ECG heartbeat patterns.")

    ## MEDIA
    # Image
    # import Image function
    from PIL import Image
    st.header("Overview on the workflow")

    # open an image
    img = Image.open("OIP (2).jpeg")

    # Plot the image
    st.image(img, caption="DataScientest")

    st.header("For More Information:")
    st.write("Explore the full original research paper at: [IEEE Xplore](https://ieeexplore.ieee.org/document/8419425)")
    st.write("Explore the original dataset at: [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)")

    
    
    
    # Audio
    #audio_file = open('name_of_file.ext', "rb")
    #audio_bytes = audio_file.read()
    #st.audio(audio_bytes, format="audio/mp3")
    
    # Video with URL
    st.subheader("une vidéo directement de YouTube:")
    st.video(data="https://www.youtube.com/watch?v=SNNK6z03TaA")
    
    ### WIDGET
    st.subheader("Let's talk about widgets")

    # Bouton
    st.button("Press ME")
    
    # getting interaction button
    if st.button("Press Me again"):
        st.success("this is a success!")
    
    # Checkbox
    if st.checkbox("Hide & seek"):
        st.success("showing")
    
    # Radio
    gender_list = ["Man", "Woman"]
    gender = st.radio("Sélectionner un genre", gender_list)
    if gender == gender_list[0]:
        st.info(f"gender is {gender}")
    
    # Select
    location = st.selectbox("Your Job", ["Data Scientist", "Dentist", "Doctor"])
    
    # Multiselect
    liste_course = st.multiselect("liste de course",
                                    ["tomates", "dentifrice", "écouteurs"])
    
    # Text imput
    name = st.text_input("your name", "your name here")
    st.text(name)
    
    # Number input
    age = st.number_input("Age", 5, 100)
    
    # text area
    message = st.text_area("Enter your message")
    
    # Slider
    niveau = st.slider("select the level", 2, 6)
    
    # Ballons
    if st.button("Press me again"):
        st.write("Yesss, you'r ready!")
        st.balloons()