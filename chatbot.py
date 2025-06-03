import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# BERT-inspired TF-IDF based chatbot
class BERTInspiredModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.knowledge_base = {}
        self.intent_vectorizer = None
        self.intent_vectors = None
        self.intents = []
        
        # Initialize with skin disease knowledge
        self._prepare_knowledge_base()
        
    def _prepare_knowledge_base(self):
        """Prepare the knowledge base with intents and responses"""
        # Define various intents for each disease and aspect
        intents = []
        
        # For each disease and aspect, create multiple phrasings
        for disease, info in SKIN_DISEASE_KB.items():
            # Description intents
            intents.append((f"what is {disease}", disease, "description"))
            intents.append((f"describe {disease}", disease, "description"))
            intents.append((f"tell me about {disease}", disease, "description"))
            
            # Symptoms intents
            intents.append((f"symptoms of {disease}", disease, "symptoms"))
            intents.append((f"what are the signs of {disease}", disease, "symptoms"))
            intents.append((f"how do I know if I have {disease}", disease, "symptoms"))
            
            # Treatment intents
            intents.append((f"how to treat {disease}", disease, "treatment"))
            intents.append((f"treatment for {disease}", disease, "treatment"))
            intents.append((f"cure for {disease}", disease, "treatment"))
            intents.append((f"medication for {disease}", disease, "treatment"))
            
            # Prevention intents
            intents.append((f"how to prevent {disease}", disease, "prevention"))
            intents.append((f"prevention of {disease}", disease, "prevention"))
            intents.append((f"avoid {disease}", disease, "prevention"))
            
            # General inquiry
            intents.append((f"tell me everything about {disease}", disease, "general"))
            
        # Add general queries
        intents.append(("what skin diseases can you tell me about", None, "list"))
        intents.append(("help", None, "help"))
        
        # Extract just the query texts
        self.intents = intents
        queries = [intent[0] for intent in intents]
        
        # Fit vectorizer on all possible queries
        self.intent_vectorizer = TfidfVectorizer().fit(queries)
        self.intent_vectors = self.intent_vectorizer.transform(queries)
        
    def find_intent(self, query):
        """Find the closest matching intent for a query"""
        query_vector = self.intent_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.intent_vectors).flatten()
        
        # Get the index of the highest similarity
        best_match_idx = np.argmax(similarities)
        similarity_score = similarities[best_match_idx]
        
        # Only accept matches above a threshold
        if similarity_score > 0.3:
            return self.intents[best_match_idx]
        else:
            return None

# Load the BERT-inspired model
@st.cache_resource
def load_bert_model():
    # Create a TF-IDF based model that simulates some BERT capabilities
    model = BERTInspiredModel()
    print("BERT-inspired chatbot model loaded successfully")
    
    return model

# Skin disease knowledge base (simplified for demonstration)
SKIN_DISEASE_KB = {
    "acne": {
        "description": "Acne is a skin condition that occurs when your hair follicles become plugged with oil and dead skin cells. It causes whiteheads, blackheads or pimples.",
        "symptoms": "Whiteheads, blackheads, pimples, cysts, nodules, papules, pustules, and redness.",
        "treatment": "Topical treatments include benzoyl peroxide, salicylic acid, and retinoids. Oral medications may include antibiotics, birth control pills (for women), or isotretinoin for severe cases.",
        "prevention": "Wash your face twice daily, avoid touching your face, use oil-free products, and maintain a balanced diet."
    },
    "hyperpigmentation": {
        "description": "Hyperpigmentation is a common condition where patches of skin become darker than the surrounding skin due to excess melanin production.",
        "symptoms": "Darker patches of skin, often on the face, hands, and other areas exposed to the sun.",
        "treatment": "Topical treatments include hydroquinone, kojic acid, vitamin C, and retinoids. Chemical peels, laser therapy, and intense pulsed light therapy can also help.",
        "prevention": "Use sunscreen daily, wear protective clothing, and avoid direct sun exposure."
    },
    "nail psoriasis": {
        "description": "Nail psoriasis is a manifestation of psoriasis that affects the fingernails and toenails, causing various changes in their appearance.",
        "symptoms": "Pitting, discoloration, abnormal nail growth, separation of the nail from the nail bed, and crumbling nails.",
        "treatment": "Topical treatments include corticosteroids, calcipotriol, and tazarotene. Systemic treatments may include methotrexate, cyclosporine, or biologics.",
        "prevention": "Keep nails trimmed, avoid trauma to nails, use moisturizers, and follow prescribed treatment plans."
    },
    "sjs-ten": {
        "description": "Stevens-Johnson Syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) are severe skin reactions, usually to medications, causing skin to blister and peel off.",
        "symptoms": "Skin pain, rash, blisters, fever, sore throat, fatigue, burning eyes, and extensive skin peeling.",
        "treatment": "Immediate medical attention is required. Treatment involves stopping the cause, supportive care, pain management, wound care, and possibly immunoglobulin therapy.",
        "prevention": "Avoid medications that previously caused reactions and inform all healthcare providers about any drug allergies."
    },
    "vitiligo": {
        "description": "Vitiligo is a condition in which the skin loses its pigment cells (melanocytes), resulting in discolored patches on different areas of the body.",
        "symptoms": "White patches on the skin, premature whitening of hair, loss of color in the tissues inside the mouth, and change in eye color.",
        "treatment": "Topical corticosteroids, calcineurin inhibitors, phototherapy, skin grafting, depigmentation, and sometimes surgical options.",
        "prevention": "No known prevention, but protecting skin from sun damage and avoiding physical trauma may help reduce progression."
    }
}

# Function to get response from the chatbot
def get_chatbot_response(bert_model, query):
    # Process the query to determine what's being asked
    query = query.lower()
    
    # Use the BERT-inspired model to find the matching intent
    intent = bert_model.find_intent(query)
    
    # If we found a matching intent
    if intent:
        intent_query, disease, aspect = intent
        
        # If it's a special intent
        if aspect == "list":
            return "I can provide information about these skin conditions: acne, hyperpigmentation, nail psoriasis, SJS-TEN, and vitiligo."
        elif aspect == "help":
            return "You can ask me about skin diseases like acne, hyperpigmentation, nail psoriasis, SJS-TEN, and vitiligo. Try questions like 'What is acne?', 'Symptoms of vitiligo', or 'How to treat hyperpigmentation'."
        
        # If we have a disease but no aspect or it's a general query
        if disease and (aspect == "general" or not aspect):
            return (f"Information about {disease}:\n\n"
                    f"Description: {SKIN_DISEASE_KB[disease]['description']}\n\n"
                    f"Symptoms: {SKIN_DISEASE_KB[disease]['symptoms']}\n\n"
                    f"Treatment: {SKIN_DISEASE_KB[disease]['treatment']}\n\n"
                    f"Prevention: {SKIN_DISEASE_KB[disease]['prevention']}")
        
        # If we have both disease and aspect
        if disease and aspect and aspect in SKIN_DISEASE_KB[disease]:
            return SKIN_DISEASE_KB[disease][aspect]
    
    # Fallback: check if any disease is explicitly mentioned
    for disease in SKIN_DISEASE_KB.keys():
        if disease in query:
            # Try to determine the aspect
            if any(word in query for word in ["what", "describe", "definition"]):
                return SKIN_DISEASE_KB[disease]["description"]
            elif any(word in query for word in ["symptom", "sign", "identify"]):
                return SKIN_DISEASE_KB[disease]["symptoms"]
            elif any(word in query for word in ["treat", "cure", "medication", "remedy"]):
                return SKIN_DISEASE_KB[disease]["treatment"]
            elif any(word in query for word in ["prevent", "avoid"]):
                return SKIN_DISEASE_KB[disease]["prevention"]
            else:
                # General information about the disease
                return (f"Information about {disease}:\n\n"
                        f"Description: {SKIN_DISEASE_KB[disease]['description']}\n\n"
                        f"Symptoms: {SKIN_DISEASE_KB[disease]['symptoms']}\n\n"
                        f"Treatment: {SKIN_DISEASE_KB[disease]['treatment']}\n\n"
                        f"Prevention: {SKIN_DISEASE_KB[disease]['prevention']}")
    
    # Default response if we couldn't understand the query
    return ("I can provide information about acne, hyperpigmentation, nail psoriasis, SJS-TEN, and vitiligo. "
            "Please specify which skin condition you're asking about.")
