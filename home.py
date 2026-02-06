import streamlit as st
from pathlib import Path


SIMULATION = Path.cwd()

heat_image = SIMULATION / "assets" / "images" / "Heat.png"
heat_manual = SIMULATION / "assets" / "manuals" / "Manual Book Heat-Conduction Equation.pdf"

laplace_image = SIMULATION / "assets" / "images" / "Laplace.png"
laplace_manual = SIMULATION / "assets" / "manuals" / "Manual Book Laplace.pdf"

if "show_heat" not in st.session_state:
    st.session_state.show_heat = False

if "show_laplace" not in st.session_state:
    st.session_state.show_laplace = False

col1, col2 = st.columns(2)

with col1:
    st.image(str(heat_image))
    
    if st.button("📘 Heat Manual"):
        st.session_state.show_heat = not st.session_state.show_heat   
        
    if st.session_state.show_heat:
        st.pdf(str(heat_manual))
        
    if st.button("🔥 Heat Simulation"):
        st.switch_page("heat.py")

with col2:
    st.image(str(laplace_image))

    if st.button("📗 Laplace Manual"):
        st.session_state.show_laplace = not st.session_state.show_laplace

    if st.session_state.show_laplace:
        st.pdf(str(laplace_manual))

    if st.button("🌊 Laplace Simulation"):
        st.switch_page("laplace.py")

