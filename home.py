import streamlit as st
from pathlib import Path


def show_pdf_controls(file_path, label):
    if file_path.exists():

        with open(file_path, "rb") as f:
            st.download_button(
                label=f"📥 Download {label}",
                data=f,
                file_name=file_path.name,
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.error(f"{label} not found")


SIMULATION = Path(__file__).resolve().parent

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

    if heat_image.exists():
        st.image(str(heat_image))
    else:
        st.warning("Heat image not found")

    if st.button("📘 Heat Manual", use_container_width=True):
        st.session_state.show_heat = not st.session_state.show_heat

    if st.session_state.show_heat:
        show_pdf_controls(heat_manual, "Heat Manual")

    if st.button("🔥 Heat Simulation", use_container_width=True):
        st.switch_page("heat.py")


with col2:

    if laplace_image.exists():
        st.image(str(laplace_image))
    else:
        st.warning("Laplace image not found")

    if st.button("📗 Laplace Manual", use_container_width=True):
        st.session_state.show_laplace = not st.session_state.show_laplace

    if st.session_state.show_laplace:
        show_pdf_controls(laplace_manual, "Laplace Manual")

    if st.button("🌊 Laplace Simulation", use_container_width=True):
        st.switch_page("laplace.py")
