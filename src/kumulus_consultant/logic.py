# src/kumulus_consultant/logic.py
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def generate_xai_explanation(recommendation: str, findings: Dict[str, Any]) -> str:
    # ... (the rest of the function remains the same)
    explanation_parts = []
    rec_lower = recommendation.lower()
    if "green space" in rec_lower or "park" in rec_lower:
        avg_ndvi = findings.get("average_ndvi")
        if isinstance(avg_ndvi, (int, float)) and avg_ndvi < 0.4:
            explanation_parts.append(
                f"The recommendation to improve green space is based on a low "
                f"average NDVI score of {avg_ndvi:.2f}, indicating sparse vegetation."
            )
    if "flood" in rec_lower:
        risk_level = findings.get("risk_level")
        if risk_level and risk_level.lower() == "high":
            explanation_parts.append(
                f"The flood risk assessment identified a 'High' vulnerability level, "
                f"justifying the focus on improved drainage."
            )
    if not explanation_parts:
        return (
            "This recommendation is based on a synthesis of the available "
            "geospatial data and established urban planning principles."
        )
    return (
        "This recommendation is supported by the following key findings:\n- "
        + "\n- ".join(explanation_parts)
    )

def apply_ethical_guardrail(
    recommendation: str, target_area_info: Dict[str, Any]
) -> Dict[str, Any]:
    # ... (the rest of the function remains the same)
    flags = []
    rec_lower = recommendation.lower()
    disruptive_keywords = ["relocation", "demolition", "resettlement", "clearance"]
    is_disruptive = any(keyword in rec_lower for keyword in disruptive_keywords)
    if is_disruptive:
        if target_area_info.get("is_historically_underserved"):
            flags.append(
                "The recommendation suggests a potentially disruptive action in a "
                "historically underserved community."
            )
    if flags:
        reason = (
            " ".join(flags)
            + " Human oversight and community engagement are strongly advised."
        )
        return {"flagged": True, "reason": reason}
    else:
        return {
            "flagged": False,
            "reason": "No immediate ethical concerns were flagged by the automated check.",
        }

def handle_ambiguity() -> str:
    # ... (the rest of the function remains the same)
    return (
        "I'm not sure how to proceed. Could you please clarify your request? "
        "For example, 'analyze flood risk in North Jakarta'."
    )