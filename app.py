import streamlit as st
from PyPDF2 import PdfReader

def get_pdf_text(pdf_doc):
    text=""
    pdf_reader=PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def main():
    st.set_page_config(page_title='chat cv',page_icon=":books:")
    st.header("chat with chat cv and assistant which help you to get your dream job")
    st.text_input('add any additional information')
    pdf_doc=st.file_uploader('upload your cv')
    if st.button('press here'):
        with st.spinner("processing"):
            #get the pdf
            rawtext=get_pdf_text(pdf_doc)
            st.write(rawtext)
            #preprocess the pds


                #create and embeddings and query 



if __name__=='__main__' :
    main()