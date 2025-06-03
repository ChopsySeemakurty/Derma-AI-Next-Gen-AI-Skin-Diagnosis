import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Enhanced BERT-inspired chatbot model using TF-IDF as a substitute for BERT embeddings
class BERTInspiredModel:
    def __init__(self):
        # Initialize TF-IDF vectorizer with more features for better matching
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            max_features=8000,    # Increased feature limit
            min_df=1,            # Minimum document frequency
            use_idf=True,        # Use inverse document frequency
            sublinear_tf=True    # Apply sublinear tf scaling (1 + log(tf))
        )
        self.intents = []
        self.responses = {}
        self.metadata = {}  # Store additional information about each intent
        
        # Prepare comprehensive knowledge base
        self._prepare_knowledge_base()
        
        # Fit vectorizer on intents
        self.vectorizer.fit([intent for intent in self.intents])
        
        print("Enhanced BERT-inspired chatbot model loaded successfully")
    
    def _prepare_knowledge_base(self):
        """Prepare an enhanced knowledge base with intents and responses"""
        # Acne
        self.intents.append("What is acne?")
        self.responses["What is acne?"] = (
            "Acne is a common inflammatory skin condition that occurs when hair follicles are clogged with oil and dead skin cells. "
            "It causes various lesions including whiteheads, blackheads, pimples, nodules, and cysts. Acne most commonly affects teenagers, "
            "though it can occur at any age and may continue into adulthood."
        )
        self.metadata["What is acne?"] = {
            "condition": "acne",
            "type": "definition",
            "severity": "varies"
        }
        
        self.intents.append("What causes acne?")
        self.responses["What causes acne?"] = (
            "Acne has several interconnected causes:\n\n"
            "1. Excess sebum (oil) production due to hormonal changes\n"
            "2. Buildup of dead skin cells that clog pores\n"
            "3. Bacteria (particularly P. acnes) colonization\n"
            "4. Inflammation triggered by immune response\n\n"
            "Contributing factors include hormonal fluctuations (puberty, menstruation, pregnancy), certain medications (corticosteroids, "
            "androgens, lithium), dietary factors (high-glycemic foods, dairy), stress, and genetics."
        )
        self.metadata["What causes acne?"] = {
            "condition": "acne",
            "type": "causes",
            "factors": ["hormones", "bacteria", "diet", "stress", "genetics"]
        }
        
        self.intents.append("How is acne treated?")
        self.responses["How is acne treated?"] = (
            "Acne treatment depends on severity and may include:\n\n"
            "• Topical treatments: Benzoyl peroxide (kills bacteria), retinoids (unclogs pores), salicylic acid (reduces inflammation)\n"
            "• Oral medications: Antibiotics (reduces bacteria and inflammation), combined oral contraceptives (regulates hormones), "
            "anti-androgen agents, isotretinoin (for severe cases)\n"
            "• Procedures: Light therapy, chemical peels, extraction of large cysts, steroid injections\n\n"
            "A consistent skincare routine with gentle cleansing and non-comedogenic products is also important. Treatment typically "
            "takes 4-8 weeks to show improvement and should be maintained to prevent recurrence."
        )
        self.metadata["How is acne treated?"] = {
            "condition": "acne",
            "type": "treatment",
            "approaches": ["topical", "oral", "procedural"]
        }
        
        self.intents.append("What are the types of acne?")
        self.responses["What are the types of acne?"] = (
            "Acne lesions are classified into:\n\n"
            "• Non-inflammatory: Blackheads (open comedones) and whiteheads (closed comedones)\n"
            "• Inflammatory: Papules (small red bumps), pustules (pimples with pus), nodules (large, painful, solid lumps beneath the skin), "
            "and cysts (painful, pus-filled lumps beneath the skin)\n\n"
            "Acne can also be categorized by severity:\n"
            "• Mild: Mostly comedones with few papules/pustules\n"
            "• Moderate: More papules/pustules with some nodules\n"
            "• Severe: Numerous nodules/cysts, may lead to scarring"
        )
        self.metadata["What are the types of acne?"] = {
            "condition": "acne",
            "type": "classification",
            "subtypes": ["comedonal", "inflammatory", "nodular", "cystic"]
        }
        
        self.intents.append("How can I prevent acne?")
        self.responses["How can I prevent acne?"] = (
            "To help prevent acne:\n\n"
            "1. Maintain a consistent skincare routine with gentle, non-comedogenic products\n"
            "2. Cleanse twice daily and after sweating\n"
            "3. Avoid touching your face and picking/squeezing pimples\n"
            "4. Wash hair regularly and keep it off your face\n"
            "5. Consider dietary modifications (reducing high-glycemic foods and dairy)\n"
            "6. Change pillowcases frequently\n"
            "7. Manage stress through exercise, meditation, or other techniques\n"
            "8. Use oil-free sunscreen and makeup labeled 'non-comedogenic'\n"
            "9. Consider preventive treatments if you're prone to breakouts"
        )
        self.metadata["How can I prevent acne?"] = {
            "condition": "acne",
            "type": "prevention",
            "approaches": ["skincare", "lifestyle", "diet"]
        }
        
        # Hyperpigmentation
        self.intents.append("What is hyperpigmentation?")
        self.responses["What is hyperpigmentation?"] = (
            "Hyperpigmentation is a common condition where patches of skin become darker than the surrounding areas due to excess melanin production. "
            "This darkening can affect any area of the skin and may appear as spots, patches, or larger areas of discoloration. "
            "While usually harmless, it can be a cosmetic concern for many people."
        )
        self.metadata["What is hyperpigmentation?"] = {
            "condition": "hyperpigmentation",
            "type": "definition",
            "severity": "cosmetic"
        }
        
        self.intents.append("What causes hyperpigmentation?")
        self.responses["What causes hyperpigmentation?"] = (
            "Hyperpigmentation has several causes:\n\n"
            "1. Sun exposure: UV radiation stimulates melanin production (sunspots/solar lentigines)\n"
            "2. Post-inflammatory hyperpigmentation: Darkening after skin inflammation from acne, eczema, or injury\n"
            "3. Melasma: Hormonal changes during pregnancy or from birth control pills\n"
            "4. Medical conditions: Addison's disease, hemochromatosis\n"
            "5. Medications: Certain antibiotics, antimalarials, chemotherapy drugs\n"
            "6. Age spots: Accumulated sun exposure over time\n"
            "7. Genetics: Some people are more prone to pigmentation issues"
        )
        self.metadata["What causes hyperpigmentation?"] = {
            "condition": "hyperpigmentation",
            "type": "causes",
            "factors": ["sun", "inflammation", "hormones", "medications", "genetics"]
        }
        
        self.intents.append("How is hyperpigmentation treated?")
        self.responses["How is hyperpigmentation treated?"] = (
            "Treatments for hyperpigmentation include:\n\n"
            "• Topical treatments:\n"
            "  - Hydroquinone: Lightens skin by inhibiting melanin production\n"
            "  - Retinoids: Accelerate cell turnover to fade dark spots\n"
            "  - Vitamin C: Brightens skin and reduces melanin production\n"
            "  - Kojic acid, azelaic acid, glycolic acid: Natural lightening agents\n"
            "  - Niacinamide: Reduces pigmentation and improves barrier function\n\n"
            "• Procedures:\n"
            "  - Chemical peels: Remove outer skin layers\n"
            "  - Microdermabrasion: Physical exfoliation\n"
            "  - Laser therapy: Targets melanin without damaging surrounding tissue\n"
            "  - Intense pulsed light (IPL): Reduces pigmentation over several sessions\n\n"
            "• Prevention:\n"
            "  - Sun protection is crucial (SPF 30+ daily, even on cloudy days)\n"
            "  - Wide-brimmed hats and sun-protective clothing\n\n"
            "Results typically take weeks to months. Consistent use of sun protection is essential to prevent recurrence."
        )
        self.metadata["How is hyperpigmentation treated?"] = {
            "condition": "hyperpigmentation",
            "type": "treatment",
            "approaches": ["topical", "procedural", "preventive"]
        }
        
        self.intents.append("What are the types of hyperpigmentation?")
        self.responses["What are the types of hyperpigmentation?"] = (
            "Main types of hyperpigmentation include:\n\n"
            "• Melasma: Larger patches of discoloration, typically on the face, often hormone-related\n"
            "• Solar lentigines (sun/age spots): Small, darkened patches from sun exposure, commonly on hands, face, shoulders\n"
            "• Post-inflammatory hyperpigmentation (PIH): Dark spots following inflammation from acne, eczema, injury\n"
            "• Freckles: Small brown spots that may multiply with sun exposure, often genetic\n"
            "• Acanthosis nigricans: Velvety dark patches in body folds, may signal insulin resistance\n"
            "• Drug-induced pigmentation: Discoloration from medications\n"
            "• Periorbital hyperpigmentation: Dark circles under the eyes from various causes"
        )
        self.metadata["What are the types of hyperpigmentation?"] = {
            "condition": "hyperpigmentation",
            "type": "classification",
            "subtypes": ["melasma", "solar lentigines", "PIH", "freckles", "acanthosis nigricans"]
        }
        
        # Vitiligo
        self.intents.append("What is vitiligo?")
        self.responses["What is vitiligo?"] = (
            "Vitiligo is a long-term skin condition characterized by patches of skin losing their pigment, resulting in white patches. "
            "It occurs when melanocytes, the cells responsible for skin color, are destroyed. Vitiligo can affect any area of skin, "
            "but commonly develops on the face, neck, hands, and in skin creases. It affects all races but is more noticeable in "
            "people with darker skin tones. The condition is not contagious or life-threatening, but can cause psychological distress."
        )
        self.metadata["What is vitiligo?"] = {
            "condition": "vitiligo",
            "type": "definition",
            "severity": "chronic"
        }
        
        self.intents.append("What causes vitiligo?")
        self.responses["What causes vitiligo?"] = (
            "The exact cause of vitiligo is not fully understood, but several factors are believed to contribute:\n\n"
            "1. Autoimmune disorder: The immune system mistakenly attacks and destroys melanocytes\n"
            "2. Genetic factors: About 30% of cases run in families\n"
            "3. Triggering events: Sunburn, stress, exposure to industrial chemicals\n"
            "4. Neural factors: Chemical released from nerve endings may be toxic to melanocytes\n"
            "5. Oxidative stress: Imbalance of antioxidants in the skin\n"
            "6. Self-destruction of melanocytes: Cells may self-destruct due to defects\n\n"
            "Vitiligo is often associated with other autoimmune conditions like thyroid disorders, type 1 diabetes, and Addison's disease."
        )
        self.metadata["What causes vitiligo?"] = {
            "condition": "vitiligo",
            "type": "causes",
            "factors": ["autoimmune", "genetic", "environmental", "stress"]
        }
        
        self.intents.append("How is vitiligo treated?")
        self.responses["How is vitiligo treated?"] = (
            "Vitiligo treatments aim to restore color or create uniformity of skin tone:\n\n"
            "• Medical treatments:\n"
            "  - Topical corticosteroids: Reduce inflammation\n"
            "  - Calcineurin inhibitors (tacrolimus, pimecrolimus): Affect immune response\n"
            "  - Phototherapy: UVB treatment or PUVA (psoralen + UVA)\n"
            "  - Oral or topical JAK inhibitors: Newer treatments showing promise\n"
            "  - Vitamin D analogs: May help repigmentation\n\n"
            "• Surgical options (for stable vitiligo):\n"
            "  - Skin grafting: Transplanting healthy skin to affected areas\n"
            "  - Blister grafting: Creating blisters on pigmented skin and transplanting them\n"
            "  - Melanocyte transplantation: Transferring melanocytes to affected areas\n\n"
            "• Depigmentation (for extensive vitiligo): Removing remaining pigment to create uniform appearance\n\n"
            "• Camouflage options: Makeup, self-tanners, tattooing\n\n"
            "Treatment effectiveness varies by individual, and maintaining sun protection is important during treatment."
        )
        self.metadata["How is vitiligo treated?"] = {
            "condition": "vitiligo",
            "type": "treatment",
            "approaches": ["medical", "surgical", "depigmentation", "camouflage"]
        }
        
        self.intents.append("What are the types of vitiligo?")
        self.responses["What are the types of vitiligo?"] = (
            "Vitiligo is classified based on distribution patterns:\n\n"
            "• Segmental vitiligo: Affects one segment of the body, often appears earlier in life, progresses for a limited time then stops\n"
            "• Non-segmental (generalized) vitiligo: Most common type, often symmetrical on both sides of the body\n"
            "• Acrofacial vitiligo: Affects extremities (hands, feet) and face\n"
            "• Mucosal vitiligo: Affects mucous membranes and lips\n"
            "• Universal vitiligo: Affects most of the body surface (rare)\n"
            "• Focal vitiligo: Isolated spots in a discrete area\n"
            "• Mixed vitiligo: Combination of segmental and non-segmental patterns\n\n"
            "Vitiligo is also described by its activity level: active (spreading), stable (not changing), or repigmenting (color returning)."
        )
        self.metadata["What are the types of vitiligo?"] = {
            "condition": "vitiligo",
            "type": "classification",
            "subtypes": ["segmental", "non-segmental", "acrofacial", "mucosal", "universal", "focal", "mixed"]
        }
        
        # SJS-TEN
        self.intents.append("What is SJS-TEN?")
        self.responses["What is SJS-TEN?"] = (
            "Stevens-Johnson Syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) are severe, potentially life-threatening skin reactions. "
            "They exist on a spectrum of the same condition, differentiated by the extent of skin detachment:\n\n"
            "• SJS: Less than 10% of body surface affected\n"
            "• SJS-TEN overlap: 10-30% of body surface affected\n"
            "• TEN: More than 30% of body surface affected\n\n"
            "These conditions begin with flu-like symptoms followed by a painful red or purplish rash that spreads and blisters. "
            "The top layer of skin then dies and sheds, leaving the body vulnerable to severe infection. Mucous membranes "
            "(mouth, eyes, genitals) are also commonly affected. SJS-TEN requires immediate emergency medical attention."
        )
        self.metadata["What is SJS-TEN?"] = {
            "condition": "SJS-TEN",
            "type": "definition",
            "severity": "severe"
        }
        
        self.intents.append("What causes SJS-TEN?")
        self.responses["What causes SJS-TEN?"] = (
            "SJS-TEN is primarily triggered by:\n\n"
            "1. Medications (responsible for about 80% of cases):\n"
            "   • Antibiotics: Sulfonamides, penicillins, cephalosporins\n"
            "   • Anticonvulsants: Carbamazepine, lamotrigine, phenytoin\n"
            "   • Pain relievers: NSAIDs, particularly oxicam derivatives\n"
            "   • Allopurinol (for gout)\n"
            "   • Nevirapine (HIV medication)\n\n"
            "2. Infections (less common):\n"
            "   • Mycoplasma pneumoniae\n"
            "   • Herpes simplex virus\n"
            "   • HIV\n\n"
            "3. Rarely:\n"
            "   • Vaccinations\n"
            "   • Physical agents (radiation therapy)\n"
            "   • Idiopathic (unknown cause)\n\n"
            "Genetic factors may predispose certain individuals to these reactions, particularly HLA gene variations."
        )
        self.metadata["What causes SJS-TEN?"] = {
            "condition": "SJS-TEN",
            "type": "causes",
            "factors": ["medications", "infections", "genetic predisposition"]
        }
        
        self.intents.append("How is SJS-TEN treated?")
        self.responses["How is SJS-TEN treated?"] = (
            "SJS-TEN is a medical emergency requiring immediate hospitalization. Treatment involves:\n\n"
            "1. Discontinuation of suspected causative medication(s)\n\n"
            "2. Supportive care (similar to burn treatment):\n"
            "   • Fluid and electrolyte management\n"
            "   • Temperature regulation\n"
            "   • Nutritional support\n"
            "   • Wound care and prevention of secondary infections\n"
            "   • Pain management\n"
            "   • Respiratory support if needed\n\n"
            "3. Specialized care for affected mucous membranes:\n"
            "   • Ophthalmological care to prevent long-term eye damage\n"
            "   • Oral and genital mucosa care\n\n"
            "4. Specific treatments (controversial, varying evidence):\n"
            "   • Intravenous immunoglobulin (IVIG)\n"
            "   • Systemic corticosteroids (debated)\n"
            "   • Cyclosporine\n"
            "   • TNF-alpha inhibitors\n\n"
            "Severe cases typically require treatment in specialized burn units or intensive care. Recovery can take weeks to months, "
            "and complications may persist long-term."
        )
        self.metadata["How is SJS-TEN treated?"] = {
            "condition": "SJS-TEN",
            "type": "treatment",
            "approaches": ["discontinuation", "supportive care", "specialized care", "immunomodulation"]
        }
        
        self.intents.append("What are the complications of SJS-TEN?")
        self.responses["What are the complications of SJS-TEN?"] = (
            "SJS-TEN can lead to serious complications:\n\n"
            "• Short-term complications:\n"
            "  - Sepsis (potentially life-threatening)\n"
            "  - Shock\n"
            "  - Respiratory failure\n"
            "  - Gastrointestinal bleeding\n"
            "  - Renal failure\n"
            "  - Multi-organ dysfunction\n\n"
            "• Long-term complications:\n"
            "  - Severe eye problems (dryness, scarring, visual impairment, blindness)\n"
            "  - Skin changes (pigmentation changes, scarring, chronic photosensitivity)\n"
            "  - Lung damage (bronchiolitis obliterans, chronic obstructive pulmonary disease)\n"
            "  - Oral complications (dryness, periodontal disease, taste abnormalities)\n"
            "  - Genital and urinary tract scarring\n"
            "  - Nail dystrophy\n"
            "  - Psychological trauma and PTSD\n\n"
            "The mortality rate for SJS is 5-10%, while TEN has a significantly higher rate of 30-40%. Survivors often require "
            "long-term follow-up and management of persistent complications."
        )
        self.metadata["What are the complications of SJS-TEN?"] = {
            "condition": "SJS-TEN",
            "type": "complications",
            "timeframe": ["short-term", "long-term"]
        }
        
        # Nail Psoriasis
        self.intents.append("What is nail psoriasis?")
        self.responses["What is nail psoriasis?"] = (
            "Nail psoriasis is a manifestation of psoriasis that affects the fingernails or toenails. It occurs in approximately 50% of people "
            "with skin psoriasis and up to 80% of people with psoriatic arthritis. Nail psoriasis can cause various changes to the nail "
            "appearance, including pitting (small depressions in the nail), discoloration, abnormal nail growth, thickening, and in "
            "some cases, separation of the nail from the nail bed (onycholysis). It can affect one or multiple nails and may cause pain, "
            "discomfort, and psychosocial distress due to cosmetic concerns."
        )
        self.metadata["What is nail psoriasis?"] = {
            "condition": "nail psoriasis",
            "type": "definition",
            "severity": "chronic"
        }
        
        self.intents.append("What causes nail psoriasis?")
        self.responses["What causes nail psoriasis?"] = (
            "Nail psoriasis shares the same underlying causes as skin psoriasis:\n\n"
            "1. Immune system dysfunction: The body's immune system mistakenly attacks healthy nail tissue\n"
            "2. Accelerated cell turnover: Nail cells grow and shed too quickly\n"
            "3. Inflammation: Ongoing inflammation affects the nail matrix and nail bed\n"
            "4. Genetic predisposition: Family history is a significant risk factor\n\n"
            "The specific triggers that can cause flare-ups include:\n"
            "• Stress\n"
            "• Minor nail injuries or trauma\n"
            "• Infections\n"
            "• Certain medications\n"
            "• Weather changes and dry conditions\n\n"
            "People with psoriatic arthritis have a higher likelihood of developing nail psoriasis than those with skin psoriasis alone."
        )
        self.metadata["What causes nail psoriasis?"] = {
            "condition": "nail psoriasis",
            "type": "causes",
            "factors": ["immune dysfunction", "inflammation", "genetics", "triggers"]
        }
        
        self.intents.append("How is nail psoriasis treated?")
        self.responses["How is nail psoriasis treated?"] = (
            "Treatment for nail psoriasis includes:\n\n"
            "• Topical treatments:\n"
            "  - Corticosteroids: Reduce inflammation\n"
            "  - Vitamin D analogs (calcipotriol): Slow cell growth\n"
            "  - Retinoids: Normalize cell growth\n"
            "  - Tacrolimus/pimecrolimus: Immunomodulators\n\n"
            "• Intralesional therapy:\n"
            "  - Corticosteroid injections into the nail bed or matrix\n\n"
            "• Systemic treatments (for severe cases or when skin/joint psoriasis is also present):\n"
            "  - Oral retinoids (acitretin)\n"
            "  - Methotrexate\n"
            "  - Cyclosporine\n\n"
            "• Biologic therapies:\n"
            "  - TNF-alpha inhibitors (adalimumab, etanercept)\n"
            "  - IL-17 inhibitors (secukinumab, ixekizumab)\n"
            "  - IL-23 inhibitors (guselkumab, risankizumab)\n\n"
            "• Nail care strategies:\n"
            "  - Keep nails trimmed and clean\n"
            "  - Avoid trauma and irritants\n"
            "  - Moisturize cuticles\n"
            "  - Use gentle nail products\n\n"
            "Treatment response is typically slow, with visible improvement taking 3-6 months for fingernails and up to 12 months for toenails."
        )
        self.metadata["How is nail psoriasis treated?"] = {
            "condition": "nail psoriasis",
            "type": "treatment",
            "approaches": ["topical", "intralesional", "systemic", "biologic", "nail care"]
        }
        
        self.intents.append("What are the signs of nail psoriasis?")
        self.responses["What are the signs of nail psoriasis?"] = (
            "Nail psoriasis has several characteristic signs:\n\n"
            "• Pitting: Small depressions on the nail surface (most common sign)\n"
            "• Onycholysis: Separation of the nail from the nail bed, usually starting at the tip\n"
            "• Oil drop/salmon patches: Reddish-yellow discoloration resembling a drop of oil under the nail\n"
            "• Subungual hyperkeratosis: Buildup of chalky material under the nail\n"
            "• Crumbling: Nail becomes friable and crumbles easily\n"
            "• Leukonychia: White spots in the nail\n"
            "• Beau's lines: Horizontal depressions across the nail\n"
            "• Red spots in the lunula: The white half-moon area at the base of the nail appears red\n"
            "• Splinter hemorrhages: Thin, red, or brown lines in the nail bed\n"
            "• Total nail dystrophy: Complete destruction of the nail plate\n\n"
            "These signs may appear in different combinations and severity levels."
        )
        self.metadata["What are the signs of nail psoriasis?"] = {
            "condition": "nail psoriasis",
            "type": "symptoms",
            "manifestations": ["pitting", "onycholysis", "oil drop", "hyperkeratosis", "crumbling"]
        }
        
        # General skin health
        self.intents.append("How can I maintain healthy skin?")
        self.responses["How can I maintain healthy skin?"] = (
            "For optimal skin health:\n\n"
            "1. Skincare routine:\n"
            "   • Cleanse gently twice daily with mild, pH-balanced products\n"
            "   • Moisturize daily to maintain barrier function\n"
            "   • Apply broad-spectrum SPF 30+ sunscreen daily, even in winter or cloudy days\n"
            "   • Exfoliate 1-2 times weekly (less for sensitive skin)\n\n"
            "2. Lifestyle factors:\n"
            "   • Stay hydrated by drinking 8+ glasses of water daily\n"
            "   • Eat a balanced diet rich in antioxidants, omega-3 fatty acids, vitamins C and E\n"
            "   • Get 7-9 hours of quality sleep\n"
            "   • Exercise regularly to improve circulation\n"
            "   • Manage stress through meditation, yoga, or other relaxation techniques\n"
            "   • Avoid smoking and limit alcohol consumption\n"
            "   • Minimize sun exposure during peak hours (10am-4pm)\n\n"
            "3. Environmental considerations:\n"
            "   • Use a humidifier in dry environments\n"
            "   • Shield skin from extreme weather conditions\n"
            "   • Consider air purifiers to reduce pollutant exposure\n\n"
            "4. Regular check-ups:\n"
            "   • Conduct monthly skin self-examinations\n"
            "   • See a dermatologist annually for professional evaluation\n"
            "   • Address any skin changes or concerns promptly"
        )
        self.metadata["How can I maintain healthy skin?"] = {
            "condition": "general",
            "type": "prevention",
            "approaches": ["skincare", "lifestyle", "environment", "monitoring"]
        }
    
    def find_intent(self, query):
        """Find the closest matching intent for a query with improved matching algorithm"""
        try:
            # Get vector for the user query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities with all intents
            intent_vectors = self.vectorizer.transform(self.intents)
            similarities = cosine_similarity(query_vector, intent_vectors)[0]
            
            # Find top 3 similarities
            top_indices = similarities.argsort()[-3:][::-1]
            top_similarities = [(self.intents[i], similarities[i]) for i in top_indices]
            
            # Print top matches for debugging
            print(f"Top matches for '{query}':")
            for intent, score in top_similarities:
                print(f"  {intent}: {score:.3f}")
            
            # Get the best match if above threshold
            best_match, best_score = top_similarities[0]
            
            # Check if the similarity is above threshold
            if best_score > 0.3:  # Threshold for matching
                return best_match
            else:
                return None
        except Exception as e:
            print(f"Error in intent matching: {e}")
            return None

# Load the chatbot model
@st.cache_resource
def load_enhanced_bert_model():
    # Initialize model
    model = BERTInspiredModel()
    return model

def get_enhanced_chatbot_response(bert_model, query):
    """Get response from chatbot with enhanced fallback strategy"""
    # Find matching intent
    intent = bert_model.find_intent(query)
    
    if intent:
        # We found a direct match
        response = bert_model.responses[intent]
        
        # Add metadata if available for more context
        if hasattr(bert_model, 'metadata') and intent in bert_model.metadata:
            condition = bert_model.metadata[intent].get('condition', '')
            response_type = bert_model.metadata[intent].get('type', '')
            
            # Log the matched intent for analytics (in a real system)
            print(f"Matched intent: '{intent}' [condition: {condition}, type: {response_type}]")
        
        return response
    else:
        # No direct match found, try to provide a helpful response
        # Check if query contains keywords related to skin conditions
        query_lower = query.lower()
        
        condition_keywords = {
            'acne': ["acne", "pimple", "zit", "blackhead", "whitehead", "breakout"],
            'hyperpigmentation': ["hyperpigmentation", "dark spot", "dark patch", "discoloration", "melasma"],
            'vitiligo': ["vitiligo", "white patch", "depigmentation", "loss of color"],
            'sjs-ten': ["sjs", "stevens-johnson", "toxic epidermal", "ten"],
            'nail psoriasis': ["nail psoriasis", "nail problem", "nail pitting", "nail separation"]
        }
        
        # Check for condition mentions
        detected_conditions = []
        for condition, keywords in condition_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_conditions.append(condition)
        
        # If we detected a condition, give a general response about it
        if detected_conditions:
            condition = detected_conditions[0]  # Take the first detected condition
            return (f"I noticed you're asking about {condition}. While I don't have a specific answer to your question, "
                    f"I can tell you about what {condition} is, its causes, or treatments. Could you please ask a more "
                    f"specific question about {condition}?")
        
        # Default response for no matches
        return (
            "I'm not sure I understand your question about skin conditions. You can ask me about:\n\n"
            "• Specific skin conditions like acne, hyperpigmentation, vitiligo, SJS-TEN, or nail psoriasis\n"
            "• Causes, symptoms, or treatments for these conditions\n"
            "• General skin health and prevention tips\n\n"
            "Please try rephrasing your question with more details about what you'd like to know."
        )