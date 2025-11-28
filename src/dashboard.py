import streamlit as st
from pathlib import Path
from main import assess_paper as run_universality_assessment

# -------------------- STREAMLIT APP --------------------
st.set_page_config(page_title="AI Metrics PDF Evaluator", layout="wide")

st.title("AI Metrics PDF Evaluator Dashboard")

# -------------------- FILE UPLOAD AND INPUT --------------------
st.sidebar.header("Input Files")

pdf_file = st.sidebar.file_uploader("Upload Paper PDF", type=["pdf"])
readme_url = st.sidebar.text_input(
    "GitHub README URL (optional)",
    placeholder="https://github.com/user/repo/blob/main/README.md",
    help="Enter the GitHub URL to the README.md file (blob or raw URL format accepted)"
)

if pdf_file:
    # Save PDF temporarily
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    pdf_path = temp_dir / "uploaded_paper.pdf"
    
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())
    
    st.info(f"PDF uploaded: {pdf_file.name}")
    
    if readme_url:
        st.info(f"README URL: {readme_url}")
    
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating PDF against metrics... This may take a few minutes."):
            try:
                # Set output path for the generated PDF
                output_pdf_path = temp_dir / "assessment_report.pdf"
                
                # Run the universality assessment
                run_universality_assessment(
                    pdf_path=pdf_path,
                    readme_url=readme_url if readme_url else None,
                    output_pdf_path=output_pdf_path
                )
                
                st.success("Evaluation complete!")
                
                # -------------------- DISPLAY AND DOWNLOAD REPORT --------------------
                st.header("Download Report")
                
                # Read the generated PDF
                if output_pdf_path.exists():
                    with open(output_pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name="assessment_report.pdf",
                        mime="application/pdf"
                    )
                    
                    # Also display the PDF in the app
                    st.subheader("Preview")
                    st.pdf(pdf_bytes)
                else:
                    st.error("PDF report was not generated. Please check the console for errors.")
                    
            except Exception as e:
                st.error(f"An error occurred during evaluation: {str(e)}")
                st.exception(e)
else:
    st.info("Please upload a PDF file to begin evaluation.")
