import streamlit as st
import json
from pathlib import Path
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from universality_agent_ollama import run_evaluation  # Import your existing run_evaluation function

# -------------------- STREAMLIT APP --------------------
st.set_page_config(page_title="AI Metrics PDF Evaluator", layout="wide")

st.title("AI Metrics PDF Evaluator Dashboard")

# -------------------- FILE UPLOAD --------------------
st.sidebar.header("Upload Files")

pdf_file = st.sidebar.file_uploader("Upload Paper PDF", type=["pdf"])

if pdf_file:
   
    # Save PDF temporarily
    pdf_path = Path("temp_uploaded.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())

    st.info("Files uploaded. Click below to start evaluation.")
    
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating PDF against metrics..."):
            # Run your evaluation function
            report = run_evaluation(paper_pdf_path=str(pdf_path), idx=0)
        
        st.success("Evaluation complete!")

        # -------------------- DISPLAY RESULTS --------------------
        st.header("Evaluation Results")
        
        for criterion, metrics in report["evaluation"].items():
            st.subheader(f"Criterion: {criterion}")
            for metric_name, result in metrics.items():
                status = result.get("status", "N/A")
                confidence = result.get("confidence", None)
                interpretation = result.get("interpretation", "")
                evidence = result.get("evidence_used", "")
                
                # Display metric result
                st.markdown(f"**Metric:** {metric_name}")
                st.markdown(f"**Status:** {status} | **Confidence:** {confidence}")
                
                with st.expander("Show Evidence / Interpretation"):
                    st.write("**Interpretation:**")
                    st.write(interpretation)
                    st.write("**Evidence Used:**")
                    st.write(evidence)
        
        # -------------------- DOWNLOAD REPORT AS PDF --------------------
        st.header("Download Report")
        
        def generate_pdf(report_data):
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            y = height - 50
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y, f"AI Metrics Evaluation Report")
            y -= 30
            c.setFont("Helvetica", 12)
            c.drawString(50, y, f"Paper: {pdf_file.name}")
            y -= 20
            for criterion, metrics in report_data["evaluation"].items():
                print(criterion)
                
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, f"Criterion: {criterion}")
                y -= 20
                for metric_name, result in metrics.items():
                    status = result.get("status", "N/A")
                    confidence = result.get("confidence", "")
                    interpretation = result.get("interpretation", "")
                    evidence = result.get("evidence_used", "")
                    
                    c.setFont("Helvetica-Bold", 11)
                    c.drawString(60, y, f"Metric: {metric_name} | Status: {status} | Confidence: {confidence}")
                    y -= 15
                    c.setFont("Helvetica", 10)

                    if type(interpretation) != str:
                        continue

                    # Wrap interpretation and evidence
                    for line in interpretation.splitlines() + evidence.splitlines():
                        if y < 50:
                            c.showPage()
                            y = height - 50
                        c.drawString(70, y, line[:95])  # truncate long lines
                        y -= 12
                    y -= 10  # space between metrics
            c.save()
            buffer.seek(0)
            return buffer

        pdf_buffer = generate_pdf(report)
        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name="universality_report.pdf",
            mime="application/pdf"
        )
