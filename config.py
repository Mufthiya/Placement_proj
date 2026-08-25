from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_MENTAL = PROJECT_ROOT / "archive (1)" / "Psychological_Assessment_Dataset.csv"
DATA_PHYSICAL = PROJECT_ROOT / "archive (3)" / "Disease_symptom_and_patient_profile_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

EPOCHS = 50
SEED = 42
TEST_SIZE = 0.2

MNLI_MODEL_NAME = "facebook/bart-large-mnli"

MENTAL_CONDITION_LABELS = [
    "generalized anxiety disorder",
    "mood disorder / depression",
    "sleep disorder",
    "stress-related condition",
    "eating disorder",
    "cognitive impairment",
    "good mental health and resilience",
]

NAVIGATION_LABELS = [
    "needs emergency care right now",
    "should see a doctor or specialist soon",
    "should start self-care and monitor symptoms",
    "is mentally and physically healthy",
]

SPECIALIST_MAP = {
    "Sleep Disorders": "Sleep medicine specialist + mental health counselor",
    "Mood Disorders": "Psychiatrist / clinical psychologist",
    "Generalized Anxiety Disorder": "Psychologist (CBT therapy) + psychiatrist review",
    "Stress-Related Conditions": "Stress-management counselor + primary care physician",
    "Eating Disorders": "Eating-disorder specialist + nutritionist",
    "Cognitive Impairments": "Neurologist + neuropsychological assessment",
    "General Mental Health": "Primary care physician (routine check-up)",
    "Coping and Resilience": "No referral needed, maintain healthy habits",
    "Post-Traumatic Stress Disorder": "Trauma-informed therapist (EMDR/CBT)",
}
