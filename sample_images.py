import streamlit as st
from PIL import Image
import io
import base64

# SVG icons for the application
# These are stored as strings to avoid using binary files

def get_acne_icon():
    """Return SVG code for acne icon."""
    return '''
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
      <circle cx="40" cy="40" r="35" fill="#FDE0DC" stroke="#E57373" stroke-width="2"/>
      <circle cx="30" cy="30" r="5" fill="#E57373"/>
      <circle cx="45" cy="55" r="4" fill="#E57373"/>
      <circle cx="55" cy="25" r="3" fill="#E57373"/>
      <circle cx="20" cy="50" r="4" fill="#E57373"/>
      <circle cx="60" cy="45" r="3" fill="#E57373"/>
    </svg>
    '''

def get_hyperpigmentation_icon():
    """Return SVG code for hyperpigmentation icon."""
    return '''
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
      <rect x="10" y="10" width="60" height="60" rx="10" fill="#FFE0B2" stroke="#8D6E63" stroke-width="2"/>
      <path d="M20,20 L30,30 M40,20 L50,30 M60,30 L70,40 M20,50 L30,60 M50,50 L60,60" stroke="#8D6E63" stroke-width="2"/>
      <circle cx="35" cy="35" r="10" fill="#8D6E63" opacity="0.6"/>
      <circle cx="50" cy="45" r="8" fill="#8D6E63" opacity="0.5"/>
      <circle cx="25" cy="50" r="6" fill="#8D6E63" opacity="0.4"/>
    </svg>
    '''

def get_nail_psoriasis_icon():
    """Return SVG code for nail psoriasis icon."""
    return '''
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
      <path d="M15,15 L65,15 C65,15 75,25 75,40 C75,55 65,65 65,65 L15,65 C15,65 5,55 5,40 C5,25 15,15 15,15 Z" fill="#FFCCBC" stroke="#FF7043" stroke-width="2"/>
      <path d="M20,20 L60,20 C60,20 70,30 70,40 C70,50 60,60 60,60 L20,60 C20,60 10,50 10,40 C10,30 20,20 20,20 Z" fill="#FBE9E7" stroke="#FF7043" stroke-width="1"/>
      <line x1="25" y1="25" x2="25" y2="55" stroke="#FF7043" stroke-width="1"/>
      <line x1="35" y1="25" x2="35" y2="55" stroke="#FF7043" stroke-width="1"/>
      <line x1="45" y1="25" x2="45" y2="55" stroke="#FF7043" stroke-width="1"/>
      <line x1="55" y1="25" x2="55" y2="55" stroke="#FF7043" stroke-width="1"/>
      <circle cx="30" cy="35" r="3" fill="#FF7043"/>
      <circle cx="40" cy="45" r="2" fill="#FF7043"/>
      <circle cx="50" cy="30" r="2" fill="#FF7043"/>
    </svg>
    '''

def get_sjs_ten_icon():
    """Return SVG code for SJS-TEN icon."""
    return '''
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
      <circle cx="40" cy="40" r="35" fill="#FFEBEE" stroke="#D32F2F" stroke-width="2"/>
      <path d="M20,20 C25,15 35,25 40,20 C45,15 55,25 60,20" stroke="#D32F2F" stroke-width="2" fill="none"/>
      <path d="M20,30 C25,25 35,35 40,30 C45,25 55,35 60,30" stroke="#D32F2F" stroke-width="2" fill="none"/>
      <path d="M20,40 C25,35 35,45 40,40 C45,35 55,45 60,40" stroke="#D32F2F" stroke-width="2" fill="none"/>
      <path d="M20,50 C25,45 35,55 40,50 C45,45 55,55 60,50" stroke="#D32F2F" stroke-width="2" fill="none"/>
      <path d="M20,60 C25,55 35,65 40,60 C45,55 55,65 60,60" stroke="#D32F2F" stroke-width="2" fill="none"/>
    </svg>
    '''

def get_vitiligo_icon():
    """Return SVG code for vitiligo icon."""
    return '''
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
      <circle cx="40" cy="40" r="35" fill="#8D6E63" stroke="#5D4037" stroke-width="2"/>
      <path d="M30,25 Q35,20 40,25 Q45,30 50,25" fill="#FFFFFF" stroke="#5D4037" stroke-width="1"/>
      <path d="M20,35 Q30,25 40,35 Q50,45 60,35" fill="#FFFFFF" stroke="#5D4037" stroke-width="1"/>
      <path d="M25,50 Q30,45 35,50 Q40,55 45,50" fill="#FFFFFF" stroke="#5D4037" stroke-width="1"/>
      <path d="M45,40 Q50,35 55,40" fill="#FFFFFF" stroke="#5D4037" stroke-width="1"/>
      <path d="M20,45 Q25,40 30,45" fill="#FFFFFF" stroke="#5D4037" stroke-width="1"/>
    </svg>
    '''

def get_chatbot_icon():
    """Return SVG code for chatbot icon."""
    return '''
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
      <rect x="10" y="10" width="60" height="50" rx="10" fill="#E3F2FD" stroke="#2196F3" stroke-width="2"/>
      <polygon points="30,60 40,70 50,60" fill="#E3F2FD" stroke="#2196F3" stroke-width="2"/>
      <circle cx="30" cy="35" r="5" fill="#2196F3"/>
      <circle cx="50" cy="35" r="5" fill="#2196F3"/>
      <path d="M25,50 Q40,60 55,50" stroke="#2196F3" stroke-width="2" fill="none"/>
    </svg>
    '''

def get_analysis_icon():
    """Return SVG code for analysis icon."""
    return '''
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
      <rect x="10" y="10" width="60" height="60" rx="5" fill="#E8F5E9" stroke="#4CAF50" stroke-width="2"/>
      <line x1="20" y1="60" x2="20" y2="20" stroke="#4CAF50" stroke-width="2"/>
      <line x1="20" y1="60" x2="60" y2="60" stroke="#4CAF50" stroke-width="2"/>
      <path d="M20,50 L30,40 L40,45 L50,30 L60,35" stroke="#4CAF50" stroke-width="2" fill="none"/>
      <circle cx="30" cy="40" r="3" fill="#4CAF50"/>
      <circle cx="40" cy="45" r="3" fill="#4CAF50"/>
      <circle cx="50" cy="30" r="3" fill="#4CAF50"/>
    </svg>
    '''

def get_report_icon():
    """Return SVG code for report icon."""
    return '''
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
      <rect x="15" y="5" width="50" height="70" rx="5" fill="#FFF3E0" stroke="#FF9800" stroke-width="2"/>
      <line x1="25" y1="20" x2="55" y2="20" stroke="#FF9800" stroke-width="2"/>
      <line x1="25" y1="30" x2="55" y2="30" stroke="#FF9800" stroke-width="2"/>
      <line x1="25" y1="40" x2="45" y2="40" stroke="#FF9800" stroke-width="2"/>
      <rect x="25" y="50" width="20" height="15" fill="#FF9800"/>
    </svg>
    '''

def get_upload_icon():
    """Return SVG code for upload icon."""
    return '''
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
      <rect x="10" y="10" width="60" height="60" rx="5" fill="#E1F5FE" stroke="#03A9F4" stroke-width="2"/>
      <path d="M40,20 L40,50" stroke="#03A9F4" stroke-width="3"/>
      <path d="M25,35 L40,20 L55,35" stroke="#03A9F4" stroke-width="3" fill="none"/>
      <path d="M20,55 L60,55" stroke="#03A9F4" stroke-width="3"/>
    </svg>
    '''

# Display icons using HTML
def display_icon(icon_svg, width=80):
    """Display an SVG icon in Streamlit."""
    html = f"""
    <div style="display: flex; justify-content: center;">
        {icon_svg}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Function to load sample SVG data as PIL images
def load_sample_svg_as_image(svg_text, size=(300, 300)):
    """Convert SVG text to PNG and load as PIL Image."""
    # We'll create a larger SVG for better quality
    svg_text = svg_text.replace('width="80"', 'width="300"')
    svg_text = svg_text.replace('height="80"', 'height="300"')
    
    # Convert SVG to PNG using PIL
    import cairosvg
    png_data = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'))
    
    # Load PNG data as PIL Image
    image = Image.open(io.BytesIO(png_data))
    
    return image
