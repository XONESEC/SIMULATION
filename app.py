import streamlit as st

st.set_page_config(page_title="Simulation Hub", layout="wide")

home_page = st.Page(
    "home.py",
    title="Home",
    icon="🏠",
    default=True
)

heat_page = st.Page(
    "heat.py",
    title="Heat Equation",
    icon="🔥"
)

laplace_page = st.Page(
    "laplace.py",
    title="Laplace Equation",
    icon="🌊"
)

st.title("Engineering Simulation Platform")

pg = st.navigation(
    {
        "Simulation Tools": [home_page, heat_page, laplace_page]
    }
)

pg.run()
