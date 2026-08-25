import json
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

from config import DATA_PHYSICAL, OUTPUT_DIR, SPECIALIST_MAP
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
from realtime_app import (
    build_self_report,
    encode_mental,
    encode_physical,
    evaluate_records,
    load_bundle,
    predict_all,
)

app = Flask(__name__)

RECORDS_FILE = OUTPUT_DIR / "ui_records.json"
BOOKINGS_FILE = OUTPUT_DIR / "ui_bookings.json"

DOCTORS = [
    {"name": "Dr. Aisha Verma", "specialty": "Psychiatry",
     "focus": "Mood & Anxiety Disorders", "days": "Mon-Fri"},
    {"name": "Dr. Daniel Okafor", "specialty": "Sleep Medicine",
     "focus": "Sleep Disorder Clinic", "days": "Tue-Sat"},
    {"name": "Dr. Emily Carter", "specialty": "Clinical Psychology",
     "focus": "CBT & Trauma Care", "days": "Mon-Thu"},
    {"name": "Dr. Raj Menon", "specialty": "Cardiology",
     "focus": "Blood Pressure & Metabolic Health", "days": "Wed-Sun"},
    {"name": "Dr. Sofia Marino", "specialty": "Nutrition Science",
     "focus": "Eating Disorder Support", "days": "Tue-Fri"},
]

URGENCY_TEXT = {
    "CRITICAL": "Based on your latest report you should see a physician immediately - within 24 hours.",
    "HIGH": "Your report flags high concern. Book a consultation within 2-3 days.",
    "MODERATE": "Some warning signs detected. A consultation within the next week is advised.",
    "LOW": "No urgent findings. A routine check-up is enough for now.",
}

PHYS_QUESTIONS = [
    {"id": "name", "label": "Your name", "type": "text",
     "placeholder": "e.g. Priya Sharma"},
    {"id": "disease", "label": "Suspected or known disease", "type": "text",
     "placeholder": "Type a disease name, e.g. Diabetes"},
    {"id": "age", "label": "Age", "type": "number", "placeholder": "e.g. 30"},
    {"id": "gender", "label": "Gender", "options": ["Female", "Male"]},
    {"id": "fever", "label": "Do you have fever?", "options": ["No", "Yes"]},
    {"id": "cough", "label": "Do you have a cough?", "options": ["No", "Yes"]},
    {"id": "fatigue", "label": "Do you feel fatigue?", "options": ["No", "Yes"]},
    {"id": "breathing", "label": "Difficulty breathing?", "options": ["No", "Yes"]},
    {"id": "bp", "label": "Blood pressure level", "options": ["Low", "Normal", "High"]},
    {"id": "cholesterol", "label": "Cholesterol level", "options": ["Low", "Normal", "High"]},
    {"id": "outcome", "label": "Doctor-diagnosed disease? (ground truth)",
     "options": ["Negative", "Positive"]},
]

MENTAL_QUESTIONS = [
    {"id": "name", "label": "Your name", "type": "text",
     "placeholder": "e.g. Priya Sharma"},
    {"id": "mood", "label": "Mood over the past two weeks", "options": MOOD_OPTIONS},
    {"id": "anxiety", "label": "Anxiety in social situations recently",
     "options": ANXIETY_OPTIONS},
    {"id": "trigger", "label": "Anxiety triggers in the past month (select all that apply)",
     "options": TRIGGER_OPTIONS, "multi": True},
    {"id": "sleep", "label": "Sleep quality over the past week", "options": SLEEP_OPTIONS},
    {"id": "appetite", "label": "Appetite changes", "options": APPETITE_OPTIONS},
    {"id": "lack_interest", "label": "Lack of interest or pleasure",
     "options": LACK_INTEREST_OPTIONS},
    {"id": "enjoyable", "label": "Engaging in enjoyable activities",
     "options": ENJOYABLE_OPTIONS},
    {"id": "phys_symptoms", "label": "Physical anxiety symptoms (palpitations, sweating)",
     "options": PHYS_SYMPTOM_OPTIONS},
    {"id": "concentration", "label": "Difficulty concentrating",
     "options": CONCENTRATION_OPTIONS},
    {"id": "coping", "label": "Coping strategies you use", "options": COPING_OPTIONS},
]

SLEEP_HOURS = {
    "Restful": 7.5, "Interrupted": 5.5, "Trouble falling asleep": 6.0,
    "Early morning waking": 6.0, "Difficulty staying asleep": 5.0,
    "None of the above": 7.0,
}

BENIGN_CONDITIONS = {"General Mental Health", "Coping and Resilience"}

_state_lock = threading.Lock()
_state = {"models_phys": None, "models_mental": None, "navigator": None,
          "phys_features": None, "mental_features": None, "mental_classes": None,
          "disease_freq": None, "error": None}


def ensure_state():
    with _state_lock:
        if _state["error"]:
            raise RuntimeError(_state["error"])
        if _state["models_phys"] is not None:
            return
        try:
            phys_summary = json.loads(
                (OUTPUT_DIR / "physical_health_summary.json").read_text())
            mental_summary = json.loads(
                (OUTPUT_DIR / "mental_health_summary.json").read_text())
            _state["phys_features"] = phys_summary["features"]
            _state["mental_features"] = mental_summary["features"]
            _state["mental_classes"] = mental_summary["classes"]
            _state["models_phys"] = load_bundle("physical_health")
            _state["models_mental"] = load_bundle("mental_health")
            df_src = pd.read_csv(DATA_PHYSICAL)
            _state["disease_freq"] = {
                k.lower(): v for k, v in
                df_src["Disease"].value_counts(normalize=True).items()}
        except Exception as exc:
            _state["error"] = str(exc)
            raise
        try:
            _state["navigator"] = MnliNavigator()
        except Exception as exc:
            print(f"MNLI unavailable: {exc}")
            _state["navigator"] = None


def load_records():
    if RECORDS_FILE.exists():
        return json.loads(RECORDS_FILE.read_text())
    return []


def save_records(records):
    OUTPUT_DIR.mkdir(exist_ok=True)
    RECORDS_FILE.write_text(json.dumps(records, indent=2))


def load_bookings():
    if BOOKINGS_FILE.exists():
        return json.loads(BOOKINGS_FILE.read_text())
    return []


def save_bookings(bookings):
    OUTPUT_DIR.mkdir(exist_ok=True)
    BOOKINGS_FILE.write_text(json.dumps(bookings, indent=2))


def extract_patient(p_ans, m_ans):
    name = (m_ans.get("name") or p_ans.get("name") or "Guest").strip() or "Guest"
    return {"name": str(name).title(), "age": p_ans.get("age"),
            "gender": p_ans.get("gender")}


def detected_issue(record):
    parts = []
    answers_p = record["answers"].get("physical", {})
    phys_pred = record["predictions"].get("physical_Random Forest")
    if phys_pred:
        disease = answers_p.get("disease", "Unknown condition")
        tag = "confirmed" if phys_pred == "Positive" else "no disease detected"
        parts.append(f"{disease} - {tag}")
    mental_pred = record["predictions"].get("mental_XGBoost")
    if mental_pred and mental_pred not in BENIGN_CONDITIONS:
        parts.append(mental_pred)
    elif mental_pred:
        parts.append("Mental state stable")
    alert = (record.get("early_warning_plan") or {}).get("alert_level")
    if alert in ("CRITICAL", "HIGH"):
        parts.append("needs prompt medical attention")
    return "; ".join(parts) if parts else "No assessment data"


def latest_record():
    records = load_records()
    return records[-1] if records else None


def criticality_of(record):
    plan = record.get("early_warning_plan") or {}
    alert = plan.get("alert_level")
    if not alert:
        phys_proba = record["probas"].get("physical_Random Forest")
        risk = phys_proba[1] if phys_proba else 0
        mental_pred = record["predictions"].get("mental_XGBoost", "")
        if risk >= 0.65 or (mental_pred and mental_pred not in BENIGN_CONDITIONS):
            alert = "MODERATE"
        else:
            alert = "LOW"
    urgency = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}[alert]
    return alert, urgency


def specialist_suggestion(record):
    cond = (record.get("early_warning_plan") or {}).get(
        "ml_predicted_mental_condition") or record["predictions"].get("mental_XGBoost")
    phys_pred = record["predictions"].get("physical_Random Forest")
    keywords = []
    if cond == "Sleep Disorders":
        keywords.append("Sleep Medicine")
    elif cond == "Eating Disorders":
        keywords.append("Nutrition Science")
    elif cond and cond not in BENIGN_CONDITIONS:
        keywords.extend(["Psychiatry", "Clinical Psychology"])
    if phys_pred == "Positive":
        keywords.append("Cardiology")
    picks = [d for d in DOCTORS if d["specialty"] in keywords]
    if not picks:
        picks = DOCTORS[:2]
    return picks


def norm_phys(raw):
    p = dict(raw)
    for key in ["fever", "cough", "fatigue", "breathing"]:
        p[key] = str(p.get(key, "No")) == "Yes"
    p["age"] = float(p.get("age", 30))
    p.setdefault("outcome", None)
    return p


def norm_mental(raw):
    m = dict(raw)
    if isinstance(m.get("trigger"), str):
        m["trigger"] = [m["trigger"]]
    m.setdefault("condition", None)
    return m


def recent_rows(latest):
    rows = []
    proba = latest["probas"].get("physical_Random Forest")
    if proba:
        risk = proba[1]
        status = ("Stable" if risk < 0.35 else
                  "Monitor" if risk < 0.65 else "Attention")
        detail = f"{round(risk * 100)}% risk"
        rows.append({"area": "Physical", "status": status, "detail": detail})
    mental_pred = latest["predictions"].get("mental_XGBoost")
    if mental_pred:
        alert = (latest.get("early_warning_plan") or {}).get("alert_level")
        if alert in ("CRITICAL", "HIGH"):
            status = "Attention"
        elif alert == "MODERATE":
            status = "Monitor"
        else:
            status = "Stable" if mental_pred in BENIGN_CONDITIONS else "Monitor"
        rows.append({"area": "Mental", "status": status, "detail": mental_pred})
    sleep_q = latest["answers"].get("mental", {}).get("sleep")
    if sleep_q:
        hours = SLEEP_HOURS.get(sleep_q)
        if hours:
            rows.append({"area": "Sleep", "status": f"{hours} hrs",
                         "detail": sleep_q})
    return rows


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dash")
def dash():
    records = load_records()
    latest = records[-1] if records else None
    return render_template("dash.html", recent=recent_rows(latest) if latest else [],
                           count=len(records), has_data=latest is not None)


@app.route("/checkin/physical")
def checkin_physical():
    try:
        ensure_state()
    except RuntimeError:
        pass
    return render_template("checkin.html", kind="physical",
                           title="Physical Check-in",
                           questions=PHYS_QUESTIONS,
                           models_error=_state["error"],
                           submit_label="RUN PHYSICAL CHECK")


@app.route("/checkin/mental")
def checkin_mental():
    try:
        ensure_state()
        classes = _state["mental_classes"]
    except RuntimeError:
        classes = []
    truth_q = {"id": "condition",
               "label": "Known mental-health diagnosis (ground truth)",
               "options": classes}
    questions = MENTAL_QUESTIONS + ([truth_q] if classes else [])
    return render_template("checkin.html", kind="mental",
                           title="Mental Well-being Check-in",
                           questions=questions,
                           models_error=_state["error"],
                           submit_label="RUN MENTAL CHECK")


@app.route("/history")
def history():
    records = list(reversed(load_records()))
    view = []
    for r in records[:30]:
        phys_proba = r["probas"].get("physical_Random Forest")
        view.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "phys": r["predictions"].get("physical_Random Forest", "-"),
            "risk": phys_proba[1] if phys_proba else None,
            "mental": r["predictions"].get("mental_XGBoost", "-"),
            "alert": (r.get("early_warning_plan") or {}).get("alert_level"),
        })
    return render_template("history.html", items=view)


@app.route("/myinfo")
def myinfo():
    records = load_records()
    phys_done = sum(1 for r in records if r["true_labels"].get("physical"))
    mental_done = sum(1 for r in records if r["true_labels"].get("mental"))
    return render_template("myinfo.html", total=len(records),
                           phys_done=phys_done, mental_done=mental_done)


@app.route("/assess")
def assess_full():
    return redirect(url_for("dash"))


@app.route("/metrics")
def metrics():
    try:
        ensure_state()
        classes = _state["mental_classes"]
    except RuntimeError:
        classes = None
    records = load_records()
    rows = evaluate_records(records, classes) if (records and classes) else []
    return render_template("metrics.html", rows=rows, count=len(records))


@app.route("/result/<rid>")
def result(rid):
    records = load_records()
    rec = next((r for r in records if r["id"] == rid), None)
    if rec is None:
        return redirect(url_for("dash"))
    rp = rec.get("results_page") or {}
    return render_template("result.html", r=rec, models=rp.get("models", []),
                           screening_top=rp.get("screening_top"),
                           plan=rp.get("plan"))


def run_section(section, raw):
    classes = _state["mental_classes"]
    record = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "answers": {},
        "predictions": {},
        "probas": {},
        "true_labels": {},
        "results_page": {"models": []},
    }
    page_models = record["results_page"]["models"]

    if section == "physical":
        p_ans = norm_phys(raw)
        px = encode_physical(p_ans, _state["phys_features"],
                             _state["disease_freq"])
        probas = predict_all(_state["models_phys"], px, 2)
        record["answers"]["physical"] = {k: str(v) for k, v in p_ans.items()}
        record["true_labels"]["physical"] = p_ans["outcome"]
        record["patient"] = extract_patient(p_ans, {})
        for model_name, pr in probas.items():
            risk = float(pr[0][1])
            pred = "Positive" if risk >= 0.5 else "Negative"
            record["predictions"][f"physical_{model_name}"] = pred
            record["probas"][f"physical_{model_name}"] = [round(float(v), 6) for v in pr[0]]
            page_models.append({
                "model": model_name, "task": "Physical outcome",
                "prediction": pred,
                "confidence": round(max(risk, 1 - risk) * 100, 1),
                "bar": round(max(risk, 1 - risk) * 100, 1)})
        return record

    m_ans = norm_mental(raw)
    mx = encode_mental(m_ans, _state["mental_features"])
    probas = predict_all(_state["models_mental"], mx, len(classes))
    record["answers"]["mental"] = m_ans
    record["true_labels"]["mental"] = m_ans["condition"]
    record["patient"] = extract_patient({}, m_ans)
    for model_name, pr in probas.items():
        idx = int(np.argmax(pr[0]))
        conf = float(pr[0][idx])
        cond = classes[idx]
        record["predictions"][f"mental_{model_name}"] = cond
        record["probas"][f"mental_{model_name}"] = [round(float(v), 6) for v in pr[0]]
        page_models.append({
            "model": model_name, "task": "Mental condition",
            "prediction": cond, "confidence": round(conf * 100, 1),
            "bar": round(conf * 100, 1)})

    navigator = _state["navigator"]
    if navigator is not None:
        text = build_self_report(m_ans)
        screening = navigator.screen_mental_state(text)
        nav_scores = navigator.recommend_action(text)
        ml_idx = int(np.argmax(probas["XGBoost"][0]))
        plan = navigator.build_early_warning(
            physical_risk_prob=None,
            mental_ml_condition=classes[ml_idx],
            mental_ml_confidence=float(np.max(probas["XGBoost"][0])),
            text_screening=screening,
            navigation_scores=nav_scores,
        )
        record["early_warning_plan"] = plan
        record["results_page"]["plan"] = plan
        record["results_page"]["screening_top"] = {
            "condition": screening["top_condition"],
            "confidence": round(screening["confidence"] * 100, 1)}
    return record


@app.route("/api/assess", methods=["POST"])
def api_assess():
    ensure_state()
    payload = request.get_json(force=True)

    if payload.get("mode") == "text":
        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Empty text"}), 400
        nav = _state["navigator"]
        if nav is None:
            return jsonify({"error": "Language model unavailable"}), 503
        screening = nav.screen_mental_state(text)
        scores = screening["scores"]
        benign = scores.get("good mental health and resilience", 0)
        concern = max((s for l, s in scores.items()
                       if l != "good mental health and resilience"), default=0)
        concern = max(0.0, concern - 0.5 * benign)
        if concern >= 0.55:
            message = "This sounds heavy. A guided mental check-in will map it properly."
            go = "mental"
        elif concern >= 0.3:
            message = "Some strain detected. A quick mental check-in could help."
            go = "mental"
        else:
            message = "Thanks for sharing. Nothing alarming jumped out today."
            go = None
        return jsonify({"top_condition": screening["top_condition"],
                        "confidence": round(screening["confidence"] * 100, 1),
                        "message": message, "go": go})

    has_phys = bool(payload.get("physical"))
    has_mental = bool(payload.get("mental"))
    if not has_phys and not has_mental:
        return jsonify({"error": "Nothing submitted"}), 400

    records = load_records()
    if has_phys and has_mental:
        p_ans = norm_phys(payload["physical"])
        m_ans = norm_mental(payload["mental"])
        px = encode_physical(p_ans, _state["phys_features"],
                             _state["disease_freq"])
        mx = encode_mental(m_ans, _state["mental_features"])
        classes = _state["mental_classes"]
        phys_probas = predict_all(_state["models_phys"], px, 2)
        mental_probas = predict_all(_state["models_mental"], mx, len(classes))
        record = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "answers": {"physical": {k: str(v) for k, v in p_ans.items()},
                        "mental": m_ans},
            "predictions": {}, "probas": {}, "true_labels": {},
            "results_page": {"models": []},
        }
        page_models = record["results_page"]["models"]
        for model_name, pr in phys_probas.items():
            risk = float(pr[0][1])
            pred = "Positive" if risk >= 0.5 else "Negative"
            record["predictions"][f"physical_{model_name}"] = pred
            record["probas"][f"physical_{model_name}"] = [round(float(v), 6) for v in pr[0]]
            page_models.append({"model": model_name, "task": "Physical outcome",
                                "prediction": pred,
                                "confidence": round(max(risk, 1 - risk) * 100, 1),
                                "bar": round(max(risk, 1 - risk) * 100, 1)})
        for model_name, pr in mental_probas.items():
            idx = int(np.argmax(pr[0]))
            conf = float(pr[0][idx])
            record["predictions"][f"mental_{model_name}"] = classes[idx]
            record["probas"][f"mental_{model_name}"] = [round(float(v), 6) for v in pr[0]]
            page_models.append({"model": model_name, "task": "Mental condition",
                                "prediction": classes[idx],
                                "confidence": round(conf * 100, 1),
                                "bar": round(conf * 100, 1)})
        navigator = _state["navigator"]
        if navigator is not None:
            text = build_self_report(m_ans)
            screening = navigator.screen_mental_state(text)
            nav_scores = navigator.recommend_action(text)
            ml_idx = int(np.argmax(mental_probas["XGBoost"][0]))
            plan = navigator.build_early_warning(
                physical_risk_prob=float(phys_probas["Random Forest"][0][1]),
                mental_ml_condition=classes[ml_idx],
                mental_ml_confidence=float(np.max(mental_probas["XGBoost"][0])),
                text_screening=screening,
                navigation_scores=nav_scores)
            record["early_warning_plan"] = plan
            record["results_page"]["plan"] = plan
            record["results_page"]["screening_top"] = {
                "condition": screening["top_condition"],
                "confidence": round(screening["confidence"] * 100, 1)}
    else:
        section = "physical" if has_phys else "mental"
        raw = payload["physical"] if has_phys else payload["mental"]
        record = run_section(section, raw)

    record.setdefault("patient", extract_patient(
        payload.get("physical") or {}, payload.get("mental") or {}))
    records.append(record)
    save_records(records)
    return jsonify({"redirect": url_for("result", rid=record["id"])})


@app.route("/api/book", methods=["POST"])
def api_book():
    payload = request.get_json(force=True)
    required = ["doctor", "date", "time", "mode"]
    missing = [k for k in required if not payload.get(k)]
    if missing:
        return jsonify({"error": f"Missing: {', '.join(missing)}"}), 400
    bookings = load_bookings()
    booking = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "created": datetime.now().isoformat(timespec="seconds"),
        "name": (payload.get("name") or "Guest").strip().title(),
        "doctor": payload["doctor"],
        "date": payload["date"],
        "time": payload["time"],
        "mode": payload["mode"],
        "contact": payload.get("contact", ""),
        "criticality": payload.get("criticality", ""),
        "reason": payload.get("reason", ""),
    }
    bookings.append(booking)
    save_bookings(bookings)
    return jsonify({"ok": True, "booking": booking})


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/reports")
def reports():
    try:
        ensure_state()
        classes = _state["mental_classes"]
    except RuntimeError:
        classes = None
    records = load_records()
    latest = records[-1] if records else None
    patient = None
    issue = None
    alert = None
    recent = []
    for r in reversed(records[-10:]):
        recent.append({
            "timestamp": r["timestamp"][:16].replace("T", " "),
            "name": r.get("patient", {}).get("name", "-"),
            "phys": r["predictions"].get("physical_Random Forest", "-"),
            "mental": r["predictions"].get("mental_XGBoost", "-"),
            "alert": (r.get("early_warning_plan") or {}).get("alert_level"),
            "issue": detected_issue(r),
        })
    if latest:
        p = latest.get("patient", {})
        patient = {
            "name": p.get("name") or "Guest",
            "age": p.get("age") or "-",
            "gender": p.get("gender") or "-",
        }
        issue = detected_issue(latest)
        alert, _ = criticality_of(latest)
    metric_rows = evaluate_records(records, classes) if (records and classes) else []
    return render_template("reports.html", patient=patient, issue=issue,
                           alert=alert, recent=recent, rows=metric_rows,
                           count=len(records))


@app.route("/appointment", methods=["GET"])
def appointment():
    latest = latest_record()
    alert = None
    urgency_text = URGENCY_TEXT["LOW"]
    specialists = DOCTORS[:2]
    patient_name = "Guest"
    reason = ""
    if latest:
        alert, _ = criticality_of(latest)
        urgency_text = URGENCY_TEXT[alert]
        specialists = specialist_suggestion(latest)
        patient_name = latest.get("patient", {}).get("name", "Guest")
        reason = detected_issue(latest)
    bookings = list(reversed(load_bookings()))[:10]
    return render_template("appointment.html", alert=alert,
                           urgency_text=urgency_text, doctors=specialists,
                           all_doctors=DOCTORS, name=patient_name,
                           reason=reason, bookings=bookings)


@app.route("/selfcare")
def selfcare():
    latest = latest_record()
    mental_pred = latest["predictions"].get("mental_XGBoost") if latest else None
    phys_pred = latest["predictions"].get("physical_Random Forest") if latest else None
    return render_template("selfcare.html",
                           mental_pred=mental_pred, phys_pred=phys_pred)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
