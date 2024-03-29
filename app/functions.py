import streamlit as st
import base64
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