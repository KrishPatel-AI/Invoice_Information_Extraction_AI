"""
Streamlit interface (bonus): upload an invoice image, run the trained
Donut checkpoint, display extracted fields + raw JSON.

Run:  streamlit run app.py -- --checkpoint /path/to/invoice_donut_final
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "05_inference"))
import streamlit as st
from PIL import Image
from infer import load_model, run_inference

st.set_page_config(page_title="Invoice Extraction", layout="wide")
st.title("Invoice Information Extraction (Donut)")

CHECKPOINT_DIR = os.environ.get("INVOICE_CHECKPOINT", "../invoice_donut_final")

@st.cache_resource
def get_model(checkpoint_dir):
    return load_model(checkpoint_dir)

checkpoint_dir = st.sidebar.text_input("Checkpoint directory", CHECKPOINT_DIR)
uploaded = st.file_uploader("Upload an invoice image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    if not os.path.isdir(checkpoint_dir):
        st.error(f"Checkpoint directory not found: {checkpoint_dir}.")
    else:
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), "_uploaded_invoice" + os.path.splitext(uploaded.name)[1])
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(tmp_path), caption="Uploaded invoice", use_container_width=True)

        with st.spinner("Running trained model..."):
            processor, model, device = get_model(checkpoint_dir)
            result = run_inference(tmp_path, processor, model, device)

        with col2:
            st.subheader("Extracted fields")
            data = result["structured_json"]
            st.write("**Invoice No:**", data.get("invoice_no"))
            st.write("**Date of issue:**", data.get("date_of_issue"))
            st.write("**Seller:**", data.get("seller", {}))
            st.write("**Client:**", data.get("client", {}))
            st.write("**Line items:**")
            st.table(data.get("items", []))
            st.write("**Total:**", data.get("total", {}))
            if not result["is_valid"]:
                st.warning(f"Schema validation issues: {result['validation_problems']}")

        st.subheader("Generated JSON")
        st.json(result["structured_json"])
        with st.expander("Raw model output (debug)"):
            st.code(result["raw_model_output"])
