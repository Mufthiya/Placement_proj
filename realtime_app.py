import argparse
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from config import DATA_PHYSICAL, OUTPUT_DIR
from data_utils import (
    ANXIETY_OPTIONS,
    APPETITE_OPTIONS,
    CONCENTRATION_OPTIONS,
    COPING_OPTIONS,
    ENJOYABLE_OPTIONS,
    LACK_INTEREST_OPTIONS,
    MOOD_OPTIONS,
    PHYS_SYMPTOM_OPTIONS,
    SLEEP_OPTIONS,
    TRIGGER_OPTIONS,
)
from mnli_module import MnliNavigator
from models_training import evaluate


def ask_choice(prompt, options):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input("Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Invalid choice, try again.")


def ask_float(prompt, lo, hi):
    while True:
        try:
            val = float(input(f"{prompt} [{lo}-{hi}]: ").strip())
            if lo <= val <= hi:
                return val
        except ValueError:
            pass
        print("Invalid value, try again.")


def load_bundle(tag):
    lin = joblib.load(OUTPUT_DIR / f"{tag}_linear.pkl")
    booster = xgb.Booster()
    booster.load_model(str(OUTPUT_DIR / f"{tag}_xgboost.json"))
    models = {"Linear Regression": lin, "XGBoost": booster}
    rf_path = OUTPUT_DIR / f"{tag}_rf.pkl"
    if rf_path.exists():
        rf = joblib.load(rf_path)
        models["Random Forest"] = rf
    return models


def xgb_proba(booster, X, num_class):
    raw = booster.predict(xgb.DMatrix(X))
    if num_class == 2:
        p1 = raw.reshape(-1, 1)
        return np.hstack([1 - p1, p1])
    return raw.reshape(-1, num_class)


def predict_all(models, X, num_class):
    out = {
        "Linear Regression": models["Linear Regression"].predict_proba(X.values),
        "XGBoost": xgb_proba(models["XGBoost"], X, num_class),
    }
    if "Random Forest" in models:
        out["Random Forest"] = models["Random Forest"].predict_proba(X.values)
    return out


def encode_physical(ans, features, disease_freq):
    row = {
        "DiseaseCode": float(disease_freq.get(ans["disease"].lower(), 0.01)),
        "Fever": float(ans["fever"]),
        "Cough": float(ans["cough"]),
        "Fatigue": float(ans["fatigue"]),
        "DifficultyBreathing": float(ans["breathing"]),
        "Age": float(ans["age"]),
        "Gender": float(0 if ans["gender"] == "Female" else 1),
        "BloodPressure": float({"Low": 0, "Normal": 1, "High": 2}[ans["bp"]]),
        "Cholesterol": float({"Low": 0, "Normal": 1, "High": 2}[ans["cholesterol"]]),
    }
    return pd.DataFrame([row]).reindex(columns=features, fill_value=0.0)


def encode_mental(ans, features):
    row = {
        "AnxietyScore": float(
            {"Not at all": 0, "Rarely anxious": 1, "Slightly anxious": 3,
             "Mildly anxious": 4, "Somewhat anxious": 5, "Fairly anxious": 6,
             "Moderately anxious": 6, "Very anxious": 8, "Extremely anxious": 9,
             "Constantly anxious": 10}[ans["anxiety"]]
        ),
        "LackInterestScore": float({"Never": 0, "Rarely": 1, "Occasionally": 2,
                                    "Frequently": 3, "Always": 4}[ans["lack_interest"]]),
        "EnjoyableScore": float({"Daily": 4, "A few times a week": 3, "Once a week": 2,
                                 "Rarely": 1, "Never": 0}[ans["enjoyable"]]),
        "ConcentrationScore": float({"Never": 0, "Occasionally": 2, "Frequently": 3,
                                     "Constantly": 4}[ans["concentration"]]),
        "PhysSymptomsScore": float({"No, not at all": 0, "Rarely": 1,
                                    "Yes, occasionally": 2,
                                    "Yes, frequently": 3}[ans["phys_symptoms"]]),
        f"Mood_{ans['mood']}": 1.0,
        f"Sleep_{ans['sleep']}": 1.0,
        f"Appetite_{ans['appetite']}": 1.0,
        f"Coping_{ans['coping']}": 1.0,
    }
    triggers = ans["trigger"]
    if isinstance(triggers, str):
        triggers = [triggers]
    for t in triggers:
        row[f"Triggers_{t}"] = 1.0
    return pd.DataFrame([row]).reindex(columns=features, fill_value=0.0)


def build_self_report(ans):
    triggers = ans["trigger"]
    if isinstance(triggers, str):
        triggers = [triggers]
    return (
        f"Mood: {ans['mood']}. Anxiety level: {ans['anxiety']}. "
        f"Triggers: {', '.join(triggers)}. Sleep: {ans['sleep']}. "
        f"Appetite: {ans['appetite']}. Loss of interest: {ans['lack_interest']}. "
        f"Enjoyable activities: {ans['enjoyable']}. "
        f"Physical symptoms: {ans['phys_symptoms']}. "
        f"Concentration: {ans['concentration']}. Coping strategies: {ans['coping']}."
    )


def collect_physical():
    print("\n--- Physical health check ---")
    return {
        "disease": input("Suspected/known disease name (e.g. Diabetes): ").strip() or "Unknown",
        "fever": ask_choice("Do you have fever?", ["No", "Yes"]) == "Yes",
        "cough": ask_choice("Do you have a cough?", ["No", "Yes"]) == "Yes",
        "fatigue": ask_choice("Do you feel fatigue?", ["No", "Yes"]) == "Yes",
        "breathing": ask_choice("Difficulty breathing?", ["No", "Yes"]) == "Yes",
        "age": ask_float("Age", 1, 120),
        "gender": ask_choice("Gender", ["Female", "Male"]),
        "bp": ask_choice("Blood pressure level", ["Low", "Normal", "High"]),
        "cholesterol": ask_choice("Cholesterol level", ["Low", "Normal", "High"]),
        "outcome": ask_choice(
            "Has a doctor diagnosed you with a disease? (ground truth)",
            ["Negative", "Positive"],
        ),
    }


def collect_mental(mental_classes):
    print("\n--- Mental well-being check ---")
    return {
        "mood": ask_choice("How would you describe your mood over the past two weeks?", MOOD_OPTIONS),
        "anxiety": ask_choice("How often have you felt anxious in social situations recently?", ANXIETY_OPTIONS),
        "trigger": ask_choice("Which anxiety triggers did you experience in the past month?", TRIGGER_OPTIONS),
        "sleep": ask_choice("How would you rate the quality of your sleep over the past week?", SLEEP_OPTIONS),
        "appetite": ask_choice("Have you noticed significant changes in appetite?", APPETITE_OPTIONS),
        "lack_interest": ask_choice("How often have you felt a lack of interest or pleasure?", LACK_INTEREST_OPTIONS),
        "enjoyable": ask_choice("How often do you engage in activities you enjoy?", ENJOYABLE_OPTIONS),
        "phys_symptoms": ask_choice("Physical symptoms of anxiety (palpitations, sweating)?", PHYS_SYMPTOM_OPTIONS),
        "concentration": ask_choice("How often do you find it difficult to concentrate?", CONCENTRATION_OPTIONS),
        "coping": ask_choice("Which coping strategies do you use when stressed?", COPING_OPTIONS),
        "condition": ask_choice("Known mental-health diagnosis (ground truth)", mental_classes),
    }


def build_feature_frames(p_ans, m_ans, phys_features, disease_freq, mental_features):
    px = encode_physical(p_ans, phys_features, disease_freq)
    mx = encode_mental(m_ans, mental_features)
    return px, mx


def normalize_ui_answers(payload):
    p = dict(payload["physical"])
    for key in ["fever", "cough", "fatigue", "breathing"]:
        p[key] = str(p.get(key, "No")) == "Yes"
    p["age"] = float(p.get("age", 30))
    return p, dict(payload["mental"])


def evaluate_records(records, mental_classes):
    if not records:
        return []
    labeled_phys = [r for r in records
                    if r["true_labels"].get("physical")]
    labeled_mental = [r for r in records
                      if r["true_labels"].get("mental")
                      and r["true_labels"]["mental"] in mental_classes]
    rows = []
    for model_name in ["Linear Regression", "Random Forest", "XGBoost"]:
        if labeled_phys:
            probas = np.array([r["probas"][f"physical_{model_name}"]
                               for r in labeled_phys])
            y = np.array([1 if r["true_labels"]["physical"] == "Positive" else 0
                          for r in labeled_phys])
            m = evaluate(y, probas, labels=[0, 1])
            rows.append({"dataset": "Physical Health", "model": model_name,
                         "accuracy": round(m["accuracy"], 4),
                         "loss": round(m["loss"], 4),
                         "precision_macro": round(m["precision_macro"], 4)})
        if labeled_mental:
            probas = np.array([r["probas"][f"mental_{model_name}"]
                               for r in labeled_mental])
            y = np.array([mental_classes.index(r["true_labels"]["mental"])
                          for r in labeled_mental])
            m = evaluate(y, probas, labels=list(range(len(mental_classes))))
            rows.append({"dataset": "Mental Health", "model": model_name,
                         "accuracy": round(m["accuracy"], 4),
                         "loss": round(m["loss"], 4),
                         "precision_macro": round(m["precision_macro"], 4)})
    return rows


DEMO_PEOPLE = [
    {
        "physical": {"disease": "Diabetes", "fever": True, "cough": False, "fatigue": True,
                     "breathing": False, "age": 52, "gender": "Male", "bp": "High",
                     "cholesterol": "High", "outcome": "Positive"},
        "mental": {"mood": "Extreme sadness", "anxiety": "Very anxious",
                   "trigger": "Work-related stress", "sleep": "Interrupted",
                   "appetite": "Loss of appetite", "lack_interest": "Always",
                   "enjoyable": "Never", "phys_symptoms": "Yes, frequently",
                   "concentration": "Constantly", "coping": "No coping strategies",
                   "condition": "Mood Disorders"},
    },
    {
        "physical": {"disease": "Common Cold", "fever": False, "cough": True, "fatigue": False,
                     "breathing": False, "age": 24, "gender": "Female", "bp": "Normal",
                     "cholesterol": "Normal", "outcome": "Negative"},
        "mental": {"mood": "Happiness", "anxiety": "Not at all",
                   "trigger": "None of the above", "sleep": "Restful",
                   "appetite": "No change", "lack_interest": "Never",
                   "enjoyable": "Daily", "phys_symptoms": "No, not at all",
                   "concentration": "Never", "coping": "Physical activity",
                   "condition": "Coping and Resilience"},
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--people", type=int, default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    phys_summary = json.loads((OUTPUT_DIR / "physical_health_summary.json").read_text())
    mental_summary = json.loads((OUTPUT_DIR / "mental_health_summary.json").read_text())
    phys_features = phys_summary["features"]
    mental_features = mental_summary["features"]
    mental_classes = mental_summary["classes"]

    phys_models = load_bundle("physical_health")
    mental_models = load_bundle("mental_health")

    df_src = pd.read_csv(DATA_PHYSICAL)
    disease_freq = {k.lower(): v for k, v in df_src["Disease"].value_counts(normalize=True).items()}

    navigator = None
    try:
        navigator = MnliNavigator()
    except Exception as exc:
        print(f"MNLI model unavailable ({exc}); continuing without text screening.")

    n_people = len(DEMO_PEOPLE) if args.demo else (
        args.people or max(1, int(input("\nHow many people will enter data now? ").strip() or 1))
    )

    records = []
    for i in range(n_people):
        print(f"\n{'=' * 60}\nPERSON {i + 1} of {n_people}\n{'=' * 60}")
        if args.demo:
            p_ans = DEMO_PEOPLE[i]["physical"]
            m_ans = DEMO_PEOPLE[i]["mental"]
            print("(demo persona auto-filled)")
        else:
            p_ans = collect_physical()
            m_ans = collect_mental(mental_classes)

        px = encode_physical(p_ans, phys_features, disease_freq)
        mx = encode_mental(m_ans, mental_features)

        phys_probas = predict_all(phys_models, px, 2)
        mental_probas = predict_all(mental_models, mx, len(mental_classes))

        record = {
            "person": i + 1,
            "answers": {"physical": {k: str(v) for k, v in p_ans.items()},
                        "mental": m_ans},
            "predictions": {},
            "probas": {},
            "true_labels": {"physical": p_ans["outcome"], "mental": m_ans["condition"]},
        }

        print("\n>>> RESULTS")
        for model_name, proba in phys_probas.items():
            risk = float(proba[0][1])
            pred = "Positive" if risk >= 0.5 else "Negative"
            record["predictions"][f"physical_{model_name}"] = pred
            record["probas"][f"physical_{model_name}"] = [round(float(v), 6) for v in proba[0]]
            print(f"[{model_name}] Physical outcome: {pred}  (risk={risk:.3f})")

        for model_name, proba in mental_probas.items():
            idx = int(np.argmax(proba[0]))
            conf = float(proba[0][idx])
            record["predictions"][f"mental_{model_name}"] = mental_classes[idx]
            record["probas"][f"mental_{model_name}"] = [round(float(v), 6) for v in proba[0]]
            print(f"[{model_name}] Mental condition: {mental_classes[idx]}  (conf={conf:.3f})")

        if navigator is not None:
            text = build_self_report(m_ans)
            screening = navigator.screen_mental_state(text)
            nav_scores = navigator.recommend_action(text)
            ml_idx = int(np.argmax(mental_probas["XGBoost"][0]))
            plan = navigator.build_early_warning(
                physical_risk_prob=float(phys_probas["Random Forest"][0][1]),
                mental_ml_condition=mental_classes[ml_idx],
                mental_ml_confidence=float(np.max(mental_probas["XGBoost"][0])),
                text_screening=screening,
                navigation_scores=nav_scores,
            )
            record["early_warning_plan"] = plan
            print(f"\nAlert level : {plan['alert_level']} (combined risk={plan['combined_risk']})")
            print(f"Specialist  : {plan['recommended_specialist']}")
            print(f"Guidance    : {plan['navigation_message']}")

        records.append(record)

    print(f"\n{'=' * 60}\nREAL-TIME MODEL EVALUATION ({len(records)} live samples)\n{'=' * 60}")

    metrics_df = pd.DataFrame(evaluate_records(records, mental_classes))
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(OUTPUT_DIR / "realtime_metrics.csv", index=False)
    with open(OUTPUT_DIR / "realtime_records.json", "w") as f:
        json.dump(records, f, indent=2)
    print("\nSaved: outputs/realtime_metrics.csv, outputs/realtime_records.json")


if __name__ == "__main__":
    main()
