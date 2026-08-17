RECOMMENDATIONS = {
    "chlorosis": {
        "mild": "Slight chlorosis detected. Likely early nitrogen deficiency or mild iron deficiency. Consider applying a diluted organic nitrogen fertilizer or liquid seaweed extract. Monitor the leaf color over the next 1-2 weeks.",
        "moderate": "Moderate chlorosis. Supplement with a balanced NPK fertilizer (e.g., 10-10-10) or blood meal. If veins remain green while tissue turns yellow (interveinal chlorosis), apply chelated iron foliar spray.",
        "severe": "Severe chlorosis. Urgent action needed. Apply a quick-release high-nitrogen liquid fertilizer. Spray leaves directly with chelated iron. Prune heavily affected yellow leaves to conserve plant energy."
    },
    "necrosis": {
        "mild": "Early signs of spot necrosis. Check soil drainage and ensure watering is consistent but not waterlogged. Test soil pH (ideal for Hibiscus is 6.0 - 6.5).",
        "moderate": "Moderate leaf tissue necrosis. Apply a multi-micronutrient spray containing manganese, zinc, and copper. Improve air circulation around the plant.",
        "severe": "Severe necrosis with significant tissue breakdown. Apply systemic micronutrient fertilizer. Prune and safely discard heavily spotted/necrosed leaves to prevent secondary opportunistic infections."
    },
    "scorch": {
        "mild": "Slight potassium deficiency showing as marginal scorch. Mix organic compost or organic wood ash into the soil. Ensure deep, uniform watering.",
        "moderate": "Moderate potassium deficiency. Add potassium sulfate or a potassium-rich fertilizer (e.g., NPK 5-5-20) to the soil. Keep soil moist.",
        "severe": "Severe marginal leaf scorch. Apply a water-soluble potassium fertilizer for quick root uptake. Trim severely dried or burnt margins to prevent further tissue tearing."
    },
    "none": {
        "none": "The leaf appears healthy and well-nourished. Continue regular watering and care. No corrective fertilizer application is needed."
    }
}

def get_recommendation(deficiency, severity):
    """
    Returns the appropriate corrective care recommendation based on deficiency type and severity.
    """
    deficiency = str(deficiency).lower()
    severity = str(severity).lower()
    
    if deficiency not in RECOMMENDATIONS:
        # Fallback to healthy
        return RECOMMENDATIONS["none"]["none"]
        
    class_recs = RECOMMENDATIONS[deficiency]
    
    if severity not in class_recs:
        # Default to mild recommendation
        default_sev = list(class_recs.keys())[0]
        return class_recs[default_sev]
        
    return class_recs[severity]
