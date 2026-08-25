import json

import pandas as pd

from config import OUTPUT_DIR, SPECIALIST_MAP
from data_utils import load_mental, load_physical_split
from mnli_module import MnliNavigator
from models_training import train_all_models


def run_pipeline(run_navigation=True):
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_final = []

    print("=" * 70)
    print("PHYSICAL HEALTH: disease symptom & patient profile dataset")
    print("=" * 70)
    X_tr, X_te, y_tr, y_te, features = load_physical_split()
    phys_models, phys_metrics, phys_proba, _ = train_all_models(
        "physical_health", X_tr, X_te, y_tr, y_te, features, multiclass=False
    )
    phys_metrics.insert(0, "dataset", "physical_health")
    all_final.append(phys_metrics)

    sample_patient = {
        "DiseaseCode": 0.01,
        "Fever": 1.0, "Cough": 1.0, "Fatigue": 1.0,
        "DifficultyBreathing": 1.0, "Age": 45.0, "Gender": 1.0,
        "BloodPressure": 2.0, "Cholesterol": 2.0,
    }
    patient_df = pd.DataFrame([sample_patient])[features]

    print()
    print("=" * 70)
    print("MENTAL HEALTH: psychological assessment dataset")
    print("=" * 70)
    mX_tr, mX_te, my_tr, my_te, m_features, text_test = load_mental()
    mental_models, mental_metrics, mental_proba, mental_class_to_idx = train_all_models(
        "mental_health", mX_tr, mX_te, my_tr, my_te, m_features, multiclass=True
    )
    mental_metrics.insert(0, "dataset", "mental_health")
    all_final.append(mental_metrics)

    combined = pd.concat(all_final, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "all_model_metrics.csv", index=False)
    idx_to_class = {i: c for c, i in mental_class_to_idx.items()}

    print()
    print("=" * 70)
    print("FINAL MODEL METRICS (epochs=50)")
    print("=" * 70)
    print(combined.to_string(index=False))

    if not run_navigation:
        return

    print()
    print("=" * 70)
    print("EARLY-WARNING + NAVIGATION DEMO (BART-large-MNLI zero-shot)")
    print("=" * 70)

    demo_text = (
        "For the past three weeks I can barely sleep at night, my heart races "
        "and I sweat a lot. I feel extremely anxious about work and family "
        "issues, I have lost interest in everything I used to enjoy, I cannot "
        "concentrate anymore and I keep skipping meals."
    )
    survey_text = str(text_test.iloc[0])

    try:
        navigator = MnliNavigator()
    except Exception as exc:
        print(f"MNLI model could not be loaded ({exc}). ML metrics above are complete.")
        return

    phys_prob_rf = float(phys_models["random_forest"].predict_proba(patient_df.values)[0][1])
    mental_probs = mental_models["xgboost"].predict_proba(mX_te.iloc[[0]].values)[0]
    top_idx = int(mental_probs.argmax())
    ml_condition = idx_to_class[top_idx]
    ml_confidence = float(mental_probs[top_idx])

    for label, text in [("free-text self-report", demo_text), ("survey respondent", survey_text)]:
        screening = navigator.screen_mental_state(text)
        navigation_scores = navigator.recommend_action(text)
        plan = navigator.build_early_warning(
            physical_risk_prob=phys_prob_rf,
            mental_ml_condition=ml_condition,
            mental_ml_confidence=ml_confidence,
            text_screening=screening,
            navigation_scores=navigation_scores,
        )
        print(f"\n--- Case: {label} ---")
        print(f"Text: {text[:160]}...")
        print(json.dumps(plan, indent=2))
        print(SPECIALIST_MAP.get(ml_condition, ""))

    with open(OUTPUT_DIR / "demo_navigation_plan.json", "w") as f:
        json.dump({"free_text_demo": plan}, f, indent=2)

    print("\nAll artifacts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    run_pipeline()
