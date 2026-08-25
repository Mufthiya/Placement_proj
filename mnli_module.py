import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    MNLI_MODEL_NAME,
    MENTAL_CONDITION_LABELS,
    NAVIGATION_LABELS,
    SPECIALIST_MAP,
)


class MnliNavigator:
    def __init__(self, model_name=MNLI_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, device_map="auto"
        )
        self.model.eval()

    @torch.no_grad()
    def zero_shot(self, text, labels):
        hypotheses = [f"This person {label}." for label in labels]
        inputs = self.tokenizer(
            [text] * len(labels),
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        entailment = probs[:, 2].cpu().numpy()
        total = entailment.sum()
        if total > 0:
            entailment = entailment / total
        ranked = sorted(zip(labels, entailment.tolist()), key=lambda kv: -kv[1])
        return dict(ranked)

    def screen_mental_state(self, text):
        scores = self.zero_shot(text, MENTAL_CONDITION_LABELS)
        top_label, top_score = next(iter(scores.items()))
        return {
            "scores": scores,
            "top_condition": top_label,
            "confidence": round(float(top_score), 4),
        }

    def recommend_action(self, text):
        scores = self.zero_shot(text, NAVIGATION_LABELS)
        return scores

    def build_early_warning(
        self,
        physical_risk_prob=None,
        mental_ml_condition=None,
        mental_ml_confidence=None,
        text_screening=None,
        navigation_scores=None,
    ):
        benign_conditions = {"General Mental Health", "Coping and Resilience"}
        ml_risk = None
        if mental_ml_confidence is not None and mental_ml_condition is not None:
            if mental_ml_condition in benign_conditions:
                ml_risk = mental_ml_confidence * 0.2
            else:
                ml_risk = mental_ml_confidence

        benign_text_label = "good mental health and resilience"
        concern_score = 0.0
        benign_text_score = 0.0
        if text_screening is not None:
            benign_text_score = text_screening["scores"].get(benign_text_label, 0.0)
            for label, score in text_screening["scores"].items():
                if label == benign_text_label:
                    continue
                concern_score = max(concern_score, score)
            concern_score = max(0.0, concern_score - 0.5 * benign_text_score)

        signals = [concern_score]
        if physical_risk_prob is not None:
            signals.append(physical_risk_prob)
        if ml_risk is not None:
            signals.append(ml_risk)
        combined_risk = max(signals)

        if combined_risk >= 0.75:
            alert = "CRITICAL"
        elif combined_risk >= 0.55:
            alert = "HIGH"
        elif combined_risk >= 0.35:
            alert = "MODERATE"
        else:
            alert = "LOW"

        specialist = SPECIALIST_MAP.get(
            mental_ml_condition, "Primary care physician"
        ) if mental_ml_condition else "Complete both check-ins for specialist routing"

        best_action = None
        if navigation_scores:
            best_action = max(navigation_scores.items(), key=lambda kv: kv[1])

        plan = {
            "alert_level": alert,
            "combined_risk": round(float(combined_risk), 4),
            "physical_risk_probability": (
                round(float(physical_risk_prob), 4)
                if physical_risk_prob is not None else None),
            "ml_predicted_mental_condition": mental_ml_condition,
            "ml_prediction_confidence": (
                round(float(mental_ml_confidence), 4)
                if mental_ml_confidence is not None else None),
            "text_detected_condition": (
                text_screening["top_condition"] if text_screening else None),
            "text_confidence": (
                round(text_screening["confidence"], 4)
                if text_screening else None),
            "text_concern_score": round(float(concern_score), 4),
            "recommended_specialist": specialist,
            "recommended_action": best_action[0] if best_action else None,
            "action_confidence": (
                round(float(best_action[1]), 4) if best_action else None),
        }

        messages = {
            "CRITICAL": (
                "Seek immediate medical attention. Contact emergency services "
                "or visit the nearest hospital."
            ),
            "HIGH": (
                "Book an appointment with a specialist within the next few days."
            ),
            "MODERATE": (
                "Schedule a consultation soon and start guided self-care while "
                "monitoring symptoms daily."
            ),
            "LOW": (
                "No urgent action needed. Keep tracking your physical and "
                "mental well-being regularly."
            ),
        }
        plan["navigation_message"] = messages[alert]
        plan["model_suggested_action"] = best_action[0]
        return plan
