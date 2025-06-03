import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
from datetime import datetime
import uuid

# Import custom modules
from models.image_classifier import load_vit_model, predict_skin_disease
from models.chatbot import load_bert_model, get_chatbot_response
from models.enhanced_chatbot import load_enhanced_bert_model, get_enhanced_chatbot_response
from utils.image_processing import preprocess_image, extract_roi
from utils.report_generator import generate_pdf_report
from utils.database import init_db, save_analysis, get_history
from utils.color_analysis import analyze_color_profile
from utils.texture_analysis import analyze_texture
from utils.metrics import display_metrics_page

# Disease information dictionary
DISEASE_INFO = {
    "Acne": {
        "description": "Inflammatory skin condition affecting hair follicles and oil glands",
        "recommendations": "Use benzoyl peroxide or salicylic acid products, keep skin clean, avoid touching face",
        "remedies": "Tea tree oil, aloe vera, zinc supplements, proper hydration"
    },
    "Hyperpigmentation": {
        "description": "Patches of skin that appear darker than surrounding areas",
        "recommendations": "Use sunscreen, avoid sun exposure, use brightening agents",
        "remedies": "Vitamin C serums, kojic acid, licorice extract, regular exfoliation"
    },
    "Nail Psoriasis": {
        "description": "Condition affecting nail growth and appearance",
        "recommendations": "Keep nails trimmed, avoid harsh chemicals, protect from trauma",
        "remedies": "Biotin supplements, tea tree oil, moisturizing treatments"
    },
    "SJS-TEN": {
        "description": "Severe skin reaction requiring immediate medical attention",
        "recommendations": "Seek emergency care immediately, stop suspected medications",
        "remedies": "Medical supervision required, no home remedies recommended"
    },
    "Vitiligo": {
        "description": "Loss of skin pigmentation in patches",
        "recommendations": "Sun protection, phototherapy under medical supervision",
        "remedies": "Vitamin D supplements, antioxidant-rich diet, stress management"
    }
}

# Set page configuration
st.set_page_config(
    page_title="Skin Disease Prediction App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define skin conditions
SKIN_CONDITIONS = ["Acne", "Hyperpigmentation", "Nail Psoriasis", "SJS-TEN", "Vitiligo"]

# Initialize session state variables if they don't exist
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_roi' not in st.session_state:
    st.session_state.current_roi = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_id' not in st.session_state:
    st.session_state.analysis_id = None

# Initialize database
init_db()

# Load models
@st.cache_resource
def load_models():
    vit_model = load_vit_model()
    bert_model = load_bert_model()
    enhanced_bert_model = load_enhanced_bert_model()
    return vit_model, bert_model, enhanced_bert_model

try:
    vit_model, bert_model, enhanced_bert_model = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    models_loaded = False

# Main app title
st.title("Skin Disease Prediction System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    app_mode = st.radio("Select a feature:", [
        "📤 Upload & Predict",
        "🔍 Advanced Analysis",
        "💬 Skin Disease Chatbot",
        "📊 Model Metrics",
        "📋 Analysis History",
        "ℹ️ About & Copyright"
    ])

    st.markdown("---")
    st.header("About")
    st.info(
        "This application uses deep learning models to analyze skin conditions. "
        "Upload an image to get started with skin disease prediction."
    )

# Upload & Predict section
if app_mode == "📤 Upload & Predict":
    st.header("Upload Image for Skin Disease Prediction")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        # Create a unique analysis ID
        if st.session_state.analysis_id is None:
            st.session_state.analysis_id = str(uuid.uuid4())

        # Read and display the image
        image = Image.open(uploaded_file)
        st.session_state.current_image = image
        col1.image(image, caption="Uploaded Image", use_container_width=True)

        # Automatically predict when image is uploaded
        if models_loaded:
            with st.spinner("Analyzing image..."):
                # Preprocess image for model
                preprocessed_img = preprocess_image(image)

                # Get prediction
                prediction, confidence_scores = predict_skin_disease(vit_model, preprocessed_img)
                st.session_state.current_prediction = {
                    "condition": SKIN_CONDITIONS[prediction],
                    "confidence": confidence_scores,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Display comprehensive analysis
                predicted_condition = SKIN_CONDITIONS[prediction]
                col2.success(f"Predicted Condition: **{predicted_condition}**")
                
                # Display confidence scores
                col2.subheader("Confidence Scores")
                fig, ax = plt.subplots()
                y_pos = np.arange(len(SKIN_CONDITIONS))
                ax.barh(y_pos, confidence_scores, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(SKIN_CONDITIONS)
                ax.invert_yaxis()
                ax.set_xlabel('Confidence Score')
                ax.set_title('Prediction Confidence')
                col2.pyplot(fig)

                # Display detailed information about the condition
                col2.subheader("Condition Information")
                col2.markdown(f"**Description:**\n{DISEASE_INFO[predicted_condition]['description']}")
                
                # Display recommendations and remedies
                col2.subheader("Recommendations")
                col2.markdown(DISEASE_INFO[predicted_condition]['recommendations'])
                
                col2.subheader("Suggested Remedies")
                col2.markdown(DISEASE_INFO[predicted_condition]['remedies'])

                # Add warning for serious conditions
                if predicted_condition == "SJS-TEN":
                    col2.error("⚠️ MEDICAL EMERGENCY: Seek immediate medical attention!")

                # Save the analysis to the database
                save_analysis(
                    analysis_id=st.session_state.analysis_id,
                    image=uploaded_file.getvalue(),
                    condition=SKIN_CONDITIONS[prediction],
                    confidence_scores=confidence_scores.tolist(),
                    timestamp=datetime.now()
                )

                # Add to history
                if st.session_state.current_prediction not in st.session_state.history:
                    st.session_state.history.append(st.session_state.current_prediction)
        else:
            st.error("Models failed to load. Please restart the application.")
            if models_loaded:
                with st.spinner("Analyzing image..."):
                    # Preprocess image for model
                    preprocessed_img = preprocess_image(image)

                    # Get prediction
                    prediction, confidence_scores = predict_skin_disease(vit_model, preprocessed_img)
                    st.session_state.current_prediction = {
                        "condition": SKIN_CONDITIONS[prediction],
                        "confidence": confidence_scores,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Display prediction results
                    col2.success(f"Predicted Condition: **{SKIN_CONDITIONS[prediction]}**")
                    col2.subheader("Confidence Scores")

                    # Create a bar chart of confidence scores
                    fig, ax = plt.subplots()
                    y_pos = np.arange(len(SKIN_CONDITIONS))
                    ax.barh(y_pos, confidence_scores, align='center')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(SKIN_CONDITIONS)
                    ax.invert_yaxis()  # Labels read top-to-bottom
                    ax.set_xlabel('Confidence Score')
                    ax.set_title('Prediction Confidence')

                    col2.pyplot(fig)

                    # Save the analysis to the database
                    save_analysis(
                        analysis_id=st.session_state.analysis_id,
                        image=uploaded_file.getvalue(),
                        condition=SKIN_CONDITIONS[prediction],
                        confidence_scores=confidence_scores.tolist(),
                        timestamp=datetime.now()
                    )

                    # Add to history
                    if st.session_state.current_prediction not in st.session_state.history:
                        st.session_state.history.append(st.session_state.current_prediction)
            else:
                st.error("Models failed to load. Please restart the application.")

# Advanced Analysis section
elif app_mode == "🔍 Advanced Analysis":
    st.header("Advanced Image Analysis")

    if st.session_state.current_image is None:
        st.warning("Please upload an image in the 'Upload & Predict' section first.")
    else:
        analysis_type = st.radio("Select Analysis Type:", [
            "Texture Analysis", 
            "Region of Interest (ROI)", 
            "Color Profile"
        ])

        if analysis_type == "Texture Analysis":
            st.subheader("Texture Analysis")
            col1, col2 = st.columns([1, 1])

            # Display original image
            col1.image(st.session_state.current_image, caption="Original Image", use_container_width=True)

            if col1.button("Analyze Texture"):
                with st.spinner("Performing texture analysis..."):
                    # Analyze texture
                    texture_results, texture_image = analyze_texture(st.session_state.current_image)

                    # Display texture analysis results
                    col2.image(texture_image, caption="Texture Visualization", use_container_width=True)

                    # Show texture metrics
                    st.subheader("Texture Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': list(texture_results.keys()),
                        'Value': list(texture_results.values())
                    })
                    st.table(metrics_df)

                    # Update database
                    if st.session_state.analysis_id:
                        save_analysis(
                            analysis_id=st.session_state.analysis_id,
                            texture_analysis=texture_results,
                            update_only=True
                        )

        elif analysis_type == "Region of Interest (ROI)":
            st.subheader("Region of Interest (ROI) Selection")
            st.info("Select a region of interest to analyze a specific area of the skin condition.")

            col1, col2 = st.columns([1, 1])

            # Display original image
            col1.image(st.session_state.current_image, caption="Original Image", use_container_width=True)

            # ROI selection
            roi_options = ["Manual Selection", "Automatic Detection"]
            roi_method = col1.radio("ROI Selection Method:", roi_options)

            if roi_method == "Manual Selection":
                # Get image dimensions for the sliders
                img_width, img_height = st.session_state.current_image.size

                # Create sliders for ROI selection
                st.subheader("Select ROI Coordinates")
                col_left, col_right = st.columns(2)

                with col_left:
                    roi_x1 = st.slider("Left X", 0, img_width, int(img_width * 0.25))
                    roi_y1 = st.slider("Top Y", 0, img_height, int(img_height * 0.25))

                with col_right:
                    roi_x2 = st.slider("Right X", 0, img_width, int(img_width * 0.75))
                    roi_y2 = st.slider("Bottom Y", 0, img_height, int(img_height * 0.75))

                if col1.button("Extract ROI"):
                    roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
                    roi_image = extract_roi(st.session_state.current_image, roi_coords, method="manual")

                    if roi_image is not None and roi_image.size[0] > 0 and roi_image.size[1] > 0:
                        st.session_state.current_roi = roi_image
                        # Display ROI
                        col2.image(roi_image, caption="Selected ROI", use_container_width=True)
                    else:
                        col2.warning("Could not extract ROI. Please adjust the selection coordinates.")

                    # Analyze the ROI
                    if models_loaded:
                        with st.spinner("Analyzing ROI..."):
                            # Preprocess ROI for model
                            preprocessed_roi = preprocess_image(roi_image)

                            # Get prediction for ROI
                            roi_prediction, roi_confidence = predict_skin_disease(vit_model, preprocessed_roi)

                            # Display ROI prediction results
                            col2.success(f"ROI Prediction: **{SKIN_CONDITIONS[roi_prediction]}**")

                            # Update database
                            if st.session_state.analysis_id:
                                save_analysis(
                                    analysis_id=st.session_state.analysis_id,
                                    roi_coordinates=roi_coords,
                                    roi_prediction=SKIN_CONDITIONS[roi_prediction],
                                    update_only=True
                                )

            else:  # Automatic Detection
                if col1.button("Auto-detect ROI"):
                    with st.spinner("Detecting regions of interest..."):
                        # Auto-detect ROI
                        roi_image, roi_coords = extract_roi(st.session_state.current_image, None, method="auto")
                        if roi_image is not None:
                            st.session_state.current_roi = roi_image
                            # Display ROI
                            col2.image(roi_image, caption="Auto-detected ROI", use_container_width=True)
                        else:
                            col2.warning("Could not detect a clear region of interest.")

                        # Analyze the ROI
                        if models_loaded and roi_image is not None:
                            # Preprocess ROI for model
                            preprocessed_roi = preprocess_image(roi_image)

                            # Get prediction for ROI
                            roi_prediction, roi_confidence = predict_skin_disease(vit_model, preprocessed_roi)

                            # Display ROI prediction results
                            col2.success(f"ROI Prediction: **{SKIN_CONDITIONS[roi_prediction]}**")

                            # Update database
                            if st.session_state.analysis_id:
                                save_analysis(
                                    analysis_id=st.session_state.analysis_id,
                                    roi_coordinates=roi_coords,
                                    roi_prediction=SKIN_CONDITIONS[roi_prediction],
                                    update_only=True
                                )

        elif analysis_type == "Color Profile":
            st.subheader("Color Profile Analysis")
            col1, col2 = st.columns([1, 1])

            # Display original image
            col1.image(st.session_state.current_image, caption="Original Image", use_container_width=True)

            if col1.button("Analyze Color Profile"):
                with st.spinner("Analyzing color profile..."):
                    # Analyze color profile
                    color_data, color_viz = analyze_color_profile(st.session_state.current_image)

                    # Display color visualization
                    col2.image(color_viz, caption="Color Segmentation", use_container_width=True)

                    # Show color distribution
                    st.subheader("Color Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = list(color_data.keys())
                    values = list(color_data.values())
                    ax.bar(colors, values, color=colors)
                    ax.set_xlabel('Color')
                    ax.set_ylabel('Percentage')
                    ax.set_title('Color Distribution in the Image')
                    st.pyplot(fig)

                    # Update database
                    if st.session_state.analysis_id:
                        save_analysis(
                            analysis_id=st.session_state.analysis_id,
                            color_analysis=color_data,
                            update_only=True
                        )

        # Generate PDF report
        if st.session_state.current_prediction:
            if st.button("Generate PDF Report"):
                with st.spinner("Generating PDF report..."):
                    # Generate PDF report
                    report_data = {
                        "image": st.session_state.current_image,
                        "prediction": st.session_state.current_prediction,
                        "texture_analysis": getattr(st.session_state, "texture_results", None),
                        "roi_image": getattr(st.session_state, "current_roi", None),
                        "color_profile": getattr(st.session_state, "color_data", None)
                    }

                    pdf_bytes = generate_pdf_report(report_data)

                    # Create download button for PDF
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"skin_analysis_{st.session_state.analysis_id}.pdf",
                        mime="application/pdf"
                    )

# Chatbot section
elif app_mode == "💬 Skin Disease Chatbot":
    st.header("Skin Disease Chatbot")

    # Add a toggle to switch between standard and enhanced chatbot
    chatbot_mode = st.radio("Select Chatbot Version:", ["Standard", "Enhanced (Knowledge-Rich)"])

    # Add image upload in chat
    chat_image = st.file_uploader("Upload an image for analysis in chat", type=["jpg", "jpeg", "png"], key="chat_image")

    # Suggested questions
    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "What is acne?",
        "How to treat hyperpigmentation?",
        "What are the symptoms of vitiligo?",
        "Is nail psoriasis contagious?",
        "What causes SJS-TEN?"
    ]

    # Create buttons for suggestions
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 3].button(suggestion):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": suggestion})

            # Generate and add response immediately
            if models_loaded:
                if chatbot_mode == "Standard":
                    response = get_chatbot_response(bert_model, suggestion)
                else:  # Enhanced mode
                    response = get_enhanced_chatbot_response(enhanced_bert_model, suggestion)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Process uploaded image if any
    if chat_image is not None:
        with st.spinner("Analyzing uploaded image..."):
            # Display the image
            image = Image.open(chat_image)
            st.image(image, caption="Uploaded Image", width=300)

            # Preprocess and predict
            preprocessed_img = preprocess_image(image)
            prediction, confidence_scores = predict_skin_disease(vit_model, preprocessed_img)

            # Add prediction to chat
            prediction_msg = f"Based on the image analysis, I detect {SKIN_CONDITIONS[prediction]} with {confidence_scores[prediction]*100:.1f}% confidence."
            st.session_state.chat_history.append({"role": "assistant", "content": prediction_msg})

    st.info(
        "Ask me any questions about skin diseases, treatments, or general skin health. "
        "For example, you can ask 'What is acne?' or 'How is vitiligo treated?'"
    )

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    user_input = st.chat_input("Ask about skin conditions...")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response based on selected chatbot version
        if models_loaded:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if chatbot_mode == "Standard":
                        response = get_chatbot_response(bert_model, user_input)
                    else:  # Enhanced mode
                        response = get_enhanced_chatbot_response(enhanced_bert_model, user_input)

                    st.write(response)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.error("Chatbot model failed to load. Please restart the application.")

# Model Metrics section
elif app_mode == "📊 Model Metrics":
    # Call the function from metrics.py that displays all the metrics visualizations
    display_metrics_page()

# Analysis History section
elif app_mode == "📋 Analysis History":
    st.header("Analysis History")

    # Get analysis history from database
    history = get_history()

    if not history:
        st.info("No previous analyses found. Upload an image to start.")
    else:
        # Display analysis history
        st.subheader("Previous Analyses")

        for i, entry in enumerate(history):
            with st.expander(f"Analysis {i+1}: {entry['condition']} - {entry['timestamp']}"):
                col1, col2 = st.columns([1, 2])

                # Convert binary image data to PIL Image
                if entry.get('image'):
                    image = Image.open(io.BytesIO(entry['image']))
                    col1.image(image, caption="Uploaded Image", use_container_width=True)

                # Display prediction details
                col2.write(f"**Predicted Condition:** {entry['condition']}")
                col2.write(f"**Analysis Date:** {entry['timestamp']}")

                # Display confidence scores if available
                if entry.get('confidence_scores'):
                    confidence_df = pd.DataFrame({
                        'Condition': SKIN_CONDITIONS,
                        'Confidence': entry['confidence_scores']
                    })
                    col2.write("**Confidence Scores:**")
                    col2.dataframe(confidence_df)

                # Display additional analyses if available
                if entry.get('texture_analysis'):
                    col2.write("**Texture Analysis Results:**")
                    col2.json(entry['texture_analysis'])

                if entry.get('color_analysis'):
                    col2.write("**Color Analysis Results:**")
                    col2.json(entry['color_analysis'])

                # Add option to regenerate report
                if st.button(f"Generate Report for Analysis {i+1}"):
                    with st.spinner("Generating PDF report..."):
                        # Create report data from history entry
                        report_data = {
                            "image": image if entry.get('image') else None,
                            "prediction": {
                                "condition": entry['condition'],
                                "confidence": entry.get('confidence_scores', []),
                                "timestamp": entry['timestamp']
                            },
                            "texture_analysis": entry.get('texture_analysis'),
                            "roi_image": None,  # ROI image not stored in history
                            "color_profile": entry.get('color_analysis')
                        }

                        pdf_bytes = generate_pdf_report(report_data)

                        # Create download button for PDF
                        st.download_button(
                            label=f"Download Report for Analysis {i+1}",
                            data=pdf_bytes,
                            file_name=f"skin_analysis_{entry['analysis_id']}.pdf",
                            mime="application/pdf"
                        )

# About & Copyright section
elif app_mode == "ℹ️ About & Copyright":
    st.header("About SkinAI")
    st.markdown("""
    ### About SkinAI

    SkinAI is an advanced skin disease prediction system that combines state-of-the-art AI models for accurate diagnosis and analysis. Our system utilizes Vision Transformers (ViT) for image analysis and BERT models for intelligent medical information retrieval.

    ### Skin Diseases We Detect

    1. **Acne**
       - Inflammatory skin condition affecting hair follicles
       - Types: Whiteheads, blackheads, papules, pustules, nodules
       - Treatment: Topical treatments, oral medications, lifestyle changes

    2. **Hyperpigmentation**
       - Darkening of skin patches due to excess melanin
       - Types: Melasma, sun spots, post-inflammatory
       - Treatment: Topical agents, chemical peels, laser therapy

    3. **Nail Psoriasis**
       - Affects nail matrix and bed
       - Signs: Pitting, discoloration, thickening, separation
       - Treatment: Topical, systemic, and biologic medications

    4. **SJS-TEN**
       - Severe skin reaction affecting multiple layers
       - Symptoms: Widespread rash, blistering, mucosal involvement
       - Treatment: Immediate medical attention, supportive care

    5. **Vitiligo**
       - Loss of skin pigmentation in patches
       - Types: Segmental, non-segmental, universal
       - Treatment: Phototherapy, topical medications, surgery

    ### AI Technology Implementation

    #### Vision Transformer (ViT)
    Our image analysis uses the ViT architecture:
    ```
    Attention(Q, K, V) = softmax(QK^T/√d_k)V

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    ```

    Key features:
    - Patch-based image processing
    - Self-attention mechanisms
    - Position embeddings
    - Multi-head attention layers

    #### BERT for Medical Language Understanding
    Our chatbot uses BERT with:
    ```
    GELU(x) = x * Φ(x)  # Activation function

    LayerNorm(x) = γ * (x - μ)/√(σ² + ε) + β
    ```

    Capabilities:
    - Contextual disease understanding
    - Symptom pattern recognition
    - Treatment recommendation matching
    - Medical terminology processing

    ### Features
    - Real-time skin disease detection
    - Advanced image analysis tools
    - Intelligent medical chatbot
    - Comprehensive reporting system
    - Secure data handling

    ### Technology Stack
    - **Image Analysis**: Vision Transformer (ViT)
    - **Natural Language Processing**: BERT
    - **Backend**: Python, Streamlit
    - **Image Processing**: OpenCV, PIL

    ### Model Architecture
    - **ViT**: 12-layer transformer with patch size 16x16
    - **BERT**: 12-layer transformer with 768 hidden dimensions
    - **Combined Inference**: Ensemble approach for accurate diagnosis

    ### Privacy & Security
    We prioritize user privacy and data security. All analyses are performed locally and data is stored securely.

    ### Copyright Notice
    © 2025 SkinAI. All rights reserved.

    This software and its content are protected by copyright law. Any unauthorized reproduction or distribution of this software, or any portion of it, may result in severe civil and criminal penalties.

    ### Contact
    For support or inquiries, please reach out to our team through the appropriate channels.

    ### Version
    Current Version: 1.0.0
    Last Updated: 2025
    """)

# Footer Section
st.markdown("---")
with st.expander("Quick Help"):
    st.markdown("""
    ### About SkinAI

    SkinAI is an advanced skin disease prediction system powered by state-of-the-art AI models:
    - **Vision Transformer (ViT)**: For accurate image-based skin disease detection
    - **BERT**: Powering our intelligent chatbot for medical information

    The system combines computer vision and natural language processing to provide:
    - Real-time skin disease analysis
    - Intelligent chat assistance
    - Detailed medical information
    - Advanced image processing

    ### Technical Stack
    - Image Analysis: Vision Transformer (ViT)
    - Natural Language Processing: BERT
    - Backend: Python, Streamlit
    - Image Processing: OpenCV, PIL

    © 2025 SkinAI. All rights reserved.
    """)

st.caption("SkinAI - Advanced Skin Disease Prediction System | Powered by ViT & BERT")