import streamlit as st
import os
from dotenv import load_dotenv


# Configure page MUST be first Streamlit command
st.set_page_config(
    page_title="DoctorAI Medical Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# Load environment variables
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    st.error("⚠️ GEMINI_API_KEY missing in .env")
    st.stop()

# Imports after page config
import io
from PIL import Image
from datetime import datetime
from utils import (
    process_file,
    analyze_image,
    save_analysis,
    get_latest_analyses,
    generate_report,
    search_pubmed,
    generate_visual_prompt_heatmap,
    process_diagnosis_file,
    generate_differential_diagnosis
)
from chat_system import render_chat_interface
from report_qa_chat import ReportQASystem
from qa_interface import render_main_qa, render_sidebar_qa

# Main
tab1, tab2, tab3 = st.tabs([
    "🔬 Medical Analysis",
    "💬 Collaboration Hub",
    "❓ Report Q&A"
])

with tab1:
    st.header("Medical Image Analysis")
    uploaded = st.file_uploader("Upload image file", type=['jpg','jpeg','png','dcm','nii','nii.gz'])
    diag_file = st.file_uploader("Upload diagnosis report (optional)", type=['pdf','txt','doc','docx','jpg','png'])
    # In your Medical Analysis tab, replace the analysis section:
if uploaded and st.button("Analyze & Generate Report"):
    # Process image
    data = process_file(uploaded)
    st.image(data["data"], caption="Original Image", use_column_width=True)

    # Process diagnosis file if uploaded
    diagnosis_text = ""
    if diag_file:
        diagnosis_text = process_diagnosis_file(diag_file)
        if diagnosis_text:
            st.success("✅ Diagnosis file processed successfully")

    # XAI Attention Heatmap with diagnosis context
    analysis = analyze_image(
        data["data"], 
        GEMINI_KEY, 
        enable_xai=True, 
        diagnosis_context=diagnosis_text  # This was missing!
    )
    
    attn = generate_visual_prompt_heatmap(data["data"], analysis["findings"], GEMINI_KEY)
    st.subheader("AI Attention Heatmap")
    st.image(attn, caption="Attention Heatmap", use_column_width=True)

    # Display summary
    st.subheader("Analysis Summary")
    st.write(analysis["analysis"])
    # --- NEW INTEGRATION ---
    st.subheader("🤖 AI-Powered Differential Diagnosis")
    with st.spinner("Calculating diagnostic probabilities..."):
        ddx_data = generate_differential_diagnosis(
            analysis["analysis"],
            analysis["findings"],
            GEMINI_KEY
        )

    if ddx_data and "diagnoses" in ddx_data:
        # Create a DataFrame for better display
        import pandas as pd
        df = pd.DataFrame(ddx_data["diagnoses"])
        df['probability'] = df['probability'] * 100 # Convert to percentage

        st.write("The AI has calculated the following diagnostic probabilities based on the available evidence:")

        # Display data in a more visually appealing way
        for index, row in df.sort_values(by='probability', ascending=False).iterrows():
            st.progress(int(row['probability']))
            st.markdown(f"**{row['condition']} ({row['probability']:.1f}%)**")
            st.markdown(f"*{row['rationale']}*")
            with st.expander("Show Supporting Evidence"):
                for evidence in row['evidence']:
                    st.info(f"- {evidence}")
    else:
        st.warning("Could not generate a probabilistic differential diagnosis.")
        if diagnosis_text:
            st.subheader("Radiologist Diagnosis")
            st.info(diagnosis_text[:500] + "..." if len(diagnosis_text) > 500 else diagnosis_text)

    # Save analysis with diagnosis
    save_analysis({
        **analysis,
        "image": data["data"],
        "heatmap": attn,
        "diagnosis": diagnosis_text,
        "filename": uploaded.name  # Add filename for chat naming
    }, uploaded.name)

    # Generate PDF report
    buf = generate_report(analysis, include_references=True, add_images=True)
    if buf:
        st.download_button(
            "⬇️ Download PDF Report",
            data=buf.getvalue(),
            file_name=f"report_{uploaded.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
    else:
        st.error("⚠️ Failed to generate PDF report.")


with tab2:
    st.write(">> Debug: entering Collaboration Hub tab")
    render_chat_interface()

with tab3:
    st.header("Report Q&A")
    if "qa_system" not in st.session_state:
        st.session_state["qa_system"] = ReportQASystem(api_key=GEMINI_KEY)
    render_sidebar_qa()
    render_main_qa()
