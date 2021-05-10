import streamlit as st
from multiapp import MultiApp
from apps import home, report,about # import your app modules here

st.set_page_config(
    page_title="F04Musician",
    layout="centered",
    initial_sidebar_state="auto",
)

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Report & Model Info", report.app)
app.add_app("About", about.app)

# The main app
app.run()
