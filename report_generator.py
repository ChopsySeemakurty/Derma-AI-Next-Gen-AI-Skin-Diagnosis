from fpdf import FPDF
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime
import streamlit as st

class SkinDiseaseReportPDF(FPDF):
    """Custom PDF class for generating skin disease analysis reports."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("helvetica", "", 12)
        
    def header(self):
        # Add company logo if available
        # self.image("logo.png", 10, 8, 33)
        
        # Set header title
        self.set_font("helvetica", "B", 18)
        self.cell(0, 10, "Skin Disease Analysis Report", 0, 1, "C")
        
        # Add date
        self.set_font("helvetica", "", 10)
        self.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "R")
        
        # Add line break
        self.ln(5)
        
    def footer(self):
        # Set position at 1.5 cm from bottom
        self.set_y(-15)
        
        # Add page number
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")
    
    def add_section_title(self, title):
        self.set_font("helvetica", "B", 14)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(2)
    
    def add_subsection_title(self, title):
        self.set_font("helvetica", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
    
    def add_text(self, text):
        self.set_font("helvetica", "", 12)
        self.multi_cell(0, 6, text)
        self.ln(2)
    
    def add_image(self, img, caption=None, w=80):
        # Save PIL image to bytes buffer
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()
        
        # Calculate x position to center the image
        x = (210 - w) / 2
        
        # Add image to PDF
        self.image(io.BytesIO(img_bytes), x=x, y=None, w=w)
        
        # Add caption if provided
        if caption:
            self.ln(2)
            self.set_font("helvetica", "I", 10)
            self.cell(0, 10, caption, 0, 1, "C")
        
        self.ln(5)
    
    def add_plot(self, fig, caption=None, w=80):
        # Save matplotlib figure to bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Calculate x position to center the image
        x = (210 - w) / 2
        
        # Add plot to PDF
        self.image(buf, x=x, y=None, w=w)
        
        # Add caption if provided
        if caption:
            self.ln(2)
            self.set_font("helvetica", "I", 10)
            self.cell(0, 10, caption, 0, 1, "C")
        
        self.ln(5)
    
    def add_table(self, headers, data):
        # Set font for headers
        self.set_font("helvetica", "B", 12)
        
        # Calculate column width
        col_width = 190 / len(headers)
        
        # Add headers
        for header in headers:
            self.cell(col_width, 10, header, 1, 0, "C")
        self.ln()
        
        # Set font for data
        self.set_font("helvetica", "", 12)
        
        # Add data rows
        for row in data:
            for item in row:
                self.cell(col_width, 10, str(item), 1, 0, "C")
            self.ln()
        
        self.ln(5)

def generate_pdf_report(report_data):
    """
    Generate a PDF report from the provided analysis data.
    
    Args:
        report_data (dict): Dictionary containing analysis data
            - image: Original uploaded image
            - prediction: Prediction results
            - texture_analysis: Results from texture analysis
            - roi_image: Region of interest image
            - color_profile: Color profile analysis results
    
    Returns:
        bytes: PDF report as bytes
    """
    # Initialize PDF
    pdf = SkinDiseaseReportPDF()
    
    # Add patient information section (placeholder)
    pdf.add_section_title("Patient Information")
    pdf.add_text("ID: Anonymous\nDate: " + datetime.now().strftime("%Y-%m-%d"))
    pdf.ln(5)
    
    # Add uploaded image section
    pdf.add_section_title("Uploaded Image")
    if report_data.get("image"):
        pdf.add_image(report_data["image"], caption="Uploaded Skin Image")
    else:
        pdf.add_text("No image available.")
    pdf.ln(5)
    
    # Add prediction results
    pdf.add_section_title("Diagnosis Results")
    if report_data.get("prediction"):
        prediction = report_data["prediction"]
        pdf.add_text(f"Predicted Condition: {prediction['condition']}")
        pdf.add_text(f"Analysis Date: {prediction.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
        
        # Add confidence scores if available
        if "confidence" in prediction and len(prediction["confidence"]) > 0:
            pdf.add_subsection_title("Confidence Scores")
            
            # Create a bar chart of confidence scores
            conditions = ["Acne", "Hyperpigmentation", "Nail Psoriasis", "SJS-TEN", "Vitiligo"]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(conditions))
            ax.barh(y_pos, prediction["confidence"], align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(conditions)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Confidence Score')
            ax.set_title('Prediction Confidence')
            
            # Add plot to PDF
            pdf.add_plot(fig, caption="Prediction Confidence Scores")
            plt.close(fig)
            
            # Add confidence scores as table
            pdf.add_subsection_title("Confidence Scores Table")
            headers = ["Condition", "Confidence"]
            data = []
            for i, condition in enumerate(conditions):
                data.append([condition, f"{prediction['confidence'][i]:.4f}"])
            pdf.add_table(headers, data)
    else:
        pdf.add_text("No prediction results available.")
    pdf.ln(5)
    
    # Add ROI analysis if available
    if report_data.get("roi_image"):
        pdf.add_section_title("Region of Interest (ROI) Analysis")
        pdf.add_image(report_data["roi_image"], caption="Selected Region of Interest")
        pdf.ln(5)
    
    # Add texture analysis if available
    if report_data.get("texture_analysis"):
        pdf.add_section_title("Texture Analysis")
        texture = report_data["texture_analysis"]
        
        # Create a table of texture metrics
        pdf.add_subsection_title("Texture Metrics")
        headers = ["Metric", "Value"]
        data = []
        for key, value in texture.items():
            data.append([key, f"{value:.4f}" if isinstance(value, float) else str(value)])
        pdf.add_table(headers, data)
        pdf.ln(5)
    
    # Add color profile analysis if available
    if report_data.get("color_profile"):
        pdf.add_section_title("Color Profile Analysis")
        color_profile = report_data["color_profile"]
        
        # Create a bar chart of color distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = list(color_profile.keys())
        values = list(color_profile.values())
        ax.bar(colors, values, color=colors)
        ax.set_xlabel('Color')
        ax.set_ylabel('Percentage')
        ax.set_title('Color Distribution in the Image')
        
        # Add plot to PDF
        pdf.add_plot(fig, caption="Color Distribution")
        plt.close(fig)
        
        # Add color distribution as table
        pdf.add_subsection_title("Color Distribution Table")
        headers = ["Color", "Percentage (%)"]
        data = []
        for color, percentage in color_profile.items():
            data.append([color, f"{percentage:.2f}"])
        pdf.add_table(headers, data)
        pdf.ln(5)
    
    # Add recommendations section
    pdf.add_section_title("Recommendations")
    if report_data.get("prediction") and "condition" in report_data["prediction"]:
        condition = report_data["prediction"]["condition"]
        
        recommendations = {
            "Acne": "1. Use gentle, non-comedogenic cleansers twice daily.\n"
                   "2. Consider topical treatments containing benzoyl peroxide or salicylic acid.\n"
                   "3. Avoid touching or picking at affected areas.\n"
                   "4. Consider dietary modifications if triggers are identified.\n"
                   "5. Consult with a dermatologist for persistent or severe acne.",
            
            "Hyperpigmentation": "1. Apply broad-spectrum sunscreen daily (SPF 30 or higher).\n"
                               "2. Consider topical treatments with vitamin C, retinoids, or kojic acid.\n"
                               "3. Avoid sun exposure, especially during peak hours.\n"
                               "4. Use physical sun protection (hats, clothing) when outdoors.\n"
                               "5. Consult with a dermatologist for persistent or severe hyperpigmentation.",
            
            "Nail Psoriasis": "1. Keep nails trimmed and clean.\n"
                            "2. Avoid trauma to nails.\n"
                            "3. Use moisturizers on nails and cuticles.\n"
                            "4. Follow prescribed topical treatments consistently.\n"
                            "5. Consult with a dermatologist for systemic treatment options if severe.",
            
            "SJS-TEN": "1. URGENT: Seek immediate medical attention - this is a serious condition.\n"
                      "2. Discontinue all non-essential medications.\n"
                      "3. Avoid self-medication.\n"
                      "4. Follow up with a dermatologist and allergist after recovery.\n"
                      "5. Maintain a list of medications that may have triggered the reaction.",
            
            "Vitiligo": "1. Apply sunscreen to protect depigmented areas from sunburn.\n"
                       "2. Consider cosmetic camouflage options.\n"
                       "3. Follow prescribed topical treatments consistently.\n"
                       "4. Protect skin from physical trauma.\n"
                       "5. Consider phototherapy options with a dermatologist."
        }
        
        if condition in recommendations:
            pdf.add_text(recommendations[condition])
        else:
            pdf.add_text("Please consult with a healthcare professional for personalized recommendations.")
    else:
        pdf.add_text("Please consult with a healthcare professional for personalized recommendations.")
    
    # Add disclaimer
    pdf.add_section_title("Disclaimer")
    pdf.add_text(
        "This analysis is provided for informational purposes only and is not intended to be a substitute "
        "for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician "
        "or other qualified health provider with any questions you may have regarding a medical condition."
    )
    
    # Get PDF as bytes
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    
    return pdf_bytes
