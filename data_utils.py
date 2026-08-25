import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_MENTAL, DATA_PHYSICAL, SEED, TEST_SIZE


def load_physical():
    df = pd.read_csv(DATA_PHYSICAL)
    df.columns = [
        "Disease", "Fever", "Cough", "Fatigue", "DifficultyBreathing",
        "Age", "Gender", "BloodPressure", "Cholesterol", "Outcome",
    ]
    yes_no = {"Yes": 1, "No": 0}
    level = {"Low": 0, "Normal": 1, "High": 2}
    df["Fever"] = df["Fever"].map(yes_no)
    df["Cough"] = df["Cough"].map(yes_no)
    df["Fatigue"] = df["Fatigue"].map(yes_no)
    df["DifficultyBreathing"] = df["DifficultyBreathing"].map(yes_no)
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
    df["BloodPressure"] = df["BloodPressure"].map(level)
    df["Cholesterol"] = df["Cholesterol"].map(level)
    disease_freq = df["Disease"].value_counts(normalize=True)
    df["DiseaseCode"] = df["Disease"].map(disease_freq)
    y = df["Outcome"].map({"Negative": 0, "Positive": 1})
    X = df.drop(columns=["Outcome", "Disease"])
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Unmapped values found in physical dataset")
    return X.astype(float), y.astype(int), list(X.columns)


MENTAL_COLUMNS = [
    "Mood", "AnxietyScale", "Triggers", "SleepQuality", "Appetite",
    "LackInterest", "Enjoyable", "PhysSymptoms", "Concentration",
    "Coping",
]

ANXIETY_ORDER = {
    "Not at all": 0, "Rarely anxious": 1, "Slightly anxious": 3,
    "Mildly anxious": 4, "Somewhat anxious": 5, "Fairly anxious": 6,
    "Moderately anxious": 6, "Very anxious": 8, "Extremely anxious": 9,
    "Constantly anxious": 10,
}

FREQUENCY_ORDER = {
    "Never": 0, "Rarely": 1, "Occasionally": 2, "A few times a week": 2,
    "Frequently": 3, "Always": 4, "Constantly": 4, "Daily": 4,
    "Once a week": 2,
}

SYMPTOM_ORDER = {
    "No, not at all": 0, "Rarely": 1, "Yes, occasionally": 2,
    "Yes, frequently": 3,
}

MOOD_OPTIONS = [
    "Happiness", "Stable", "Fluctuating", "Irritability",
    "Anxiety", "Mild sadness", "Extreme sadness",
]

TRIGGER_OPTIONS = [
    "None of the above", "Work-related stress", "Family issues",
    "Financial concerns", "Health concerns", "Social situations",
]

SLEEP_OPTIONS = [
    "Restful", "Interrupted", "Trouble falling asleep",
    "Early morning waking", "Difficulty staying asleep",
    "None of the above",
]

APPETITE_OPTIONS = [
    "No change", "Increased cravings", "Loss of appetite",
    "Fluctuates daily",
]

COPING_OPTIONS = [
    "Physical activity", "Journaling or writing",
    "Mindfulness or meditation", "Social engagement",
    "No coping strategies",
]

LACK_INTEREST_OPTIONS = ["Never", "Rarely", "Occasionally", "Frequently", "Always"]

ENJOYABLE_OPTIONS = ["Daily", "A few times a week", "Once a week", "Rarely", "Never"]

CONCENTRATION_OPTIONS = ["Never", "Occasionally", "Frequently", "Constantly"]

PHYS_SYMPTOM_OPTIONS = list(SYMPTOM_ORDER.keys())

ANXIETY_OPTIONS = list(ANXIETY_ORDER.keys())


def load_mental():
    df = pd.read_csv(DATA_MENTAL)
    df.columns = MENTAL_COLUMNS + ["Condition"]
    df["Condition"] = df["Condition"].astype(str).str.split(":").str[0].str.strip()

    df["AnxietyScore"] = df["AnxietyScale"].map(ANXIETY_ORDER).fillna(5)
    df["LackInterestScore"] = df["LackInterest"].map(FREQUENCY_ORDER)
    df["EnjoyableScore"] = df["Enjoyable"].map(FREQUENCY_ORDER)
    df["ConcentrationScore"] = df["Concentration"].map(FREQUENCY_ORDER)
    df["PhysSymptomsScore"] = df["PhysSymptoms"].map(SYMPTOM_ORDER)

    ordinal = df[[
        "AnxietyScore", "LackInterestScore", "EnjoyableScore",
        "ConcentrationScore", "PhysSymptomsScore",
    ]].astype(float)

    nominal = pd.get_dummies(
        df[["Mood", "Triggers", "SleepQuality", "Appetite", "Coping"]],
        prefix=["Mood", "Triggers", "Sleep", "Appetite", "Coping"],
    ).astype(float)

    X = pd.concat([ordinal, nominal], axis=1)
    y = df["Condition"]

    risk_text = (
        "Mood: " + df["Mood"].astype(str)
        + ". Anxiety level: " + df["AnxietyScale"].astype(str)
        + ". Triggers: " + df["Triggers"].astype(str)
        + ". Sleep: " + df["SleepQuality"].astype(str)
        + ". Appetite: " + df["Appetite"].astype(str)
        + ". Loss of interest: " + df["LackInterest"].astype(str)
        + ". Enjoyable activities: " + df["Enjoyable"].astype(str)
        + ". Physical symptoms: " + df["PhysSymptoms"].astype(str)
        + ". Concentration: " + df["Concentration"].astype(str)
        + ". Coping strategies: " + df["Coping"].astype(str) + "."
    )

    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
        X, y, risk_text, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test, list(X.columns), text_test.reset_index(drop=True)


def load_physical_split():
    X, y, feature_names = load_physical()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names
