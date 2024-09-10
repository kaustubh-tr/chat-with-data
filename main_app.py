import streamlit as st
from chat_with_pdf import main as chat_with_pdf_main
from chat_with_txt import main as chat_with_txt_main
from chat_with_csv import main as chat_with_csv_main
from chat_with_sql import main as chat_with_sql_main
from chat_with_website import main as chat_with_website_main

st.set_page_config(page_title="Chat with Data", layout="wide")  # <--- Call set_page_config() here
st.markdown("<h1 style='text-align: center;'>Chat Web App</h1>", unsafe_allow_html=True)
# st.divider()

options = {
    "Chat with PDF": chat_with_pdf_main,
    "Chat with TXT": chat_with_txt_main,
    "Chat with CSV": chat_with_csv_main,
    "Chat with SQL": chat_with_sql_main,
    "Chat with Website": chat_with_website_main
}

selection = st.selectbox("Select an option:", list(options.keys()))

if selection:
    options[selection]()