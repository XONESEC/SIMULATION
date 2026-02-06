import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

heat_image = BASE_DIR / "assets" / "images" / "Heat.png"
heat_manual = BASE_DIR / "assets" / "manuals" / "Manual Book Heat-Conduction Equation.pdf"

laplace_image = BASE_DIR / "assets" / "images" / "Laplace.png"
laplace_manual = BASE_DIR / "assets" / "manuals" / "Manual Book Laplace.pdf"


if "show_heat" not in st.session_state:
    st.session_state.show_heat = False

if "show_laplace" not in st.session_state:
    st.session_state.show_laplace = False

col1, col2 = st.columns(2)


with col1:

    if heat_image.exists():
        st.image(str(heat_image))
    else:
        st.warning("Heat image not found")

    if st.button("📘 Heat Manual"):
        st.session_state.show_heat = not st.session_state.show_heat

    if st.session_state.show_heat:
        if heat_manual.exists():
            st.pdf(str(heat_manual))
        else:
            st.error("Heat manual PDF not found")

    if st.button("🔥 Heat Simulation"):
        st.switch_page("heat.py")

with col2:

    if laplace_image.exists():
        st.image(str(laplace_image))
    else:
        st.warning("Laplace image not found")

    if st.button("📗 Laplace Manual"):
        st.session_state.show_laplace = not st.session_state.show_laplace

    if st.session_state.show_laplace:
        if laplace_manual.exists():
            st.pdf(str(laplace_manual))
        else:
            st.error("Laplace manual PDF not found")

    if st.button("🌊 Laplace Simulation"):
        st.switch_page("laplace.py")
