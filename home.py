import streamlit as st
from pathlib import Path
import base64

def show_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}"
        width="100%" height="600" type="application/pdf"></iframe>
        """

        st.markdown(pdf_display, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Gagal membuka PDF: {e}")


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

    if st.button("📘 Heat Manual"):
        st.session_state.show_heat = not st.session_state.show_heat

    if st.session_state.show_heat:
        if heat_manual.exists():
            show_pdf(heat_manual)
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
            show_pdf(laplace_manual)
        else:
            st.error("Laplace manual PDF not found")

    if st.button("🌊 Laplace Simulation"):
        st.switch_page("laplace.py")



