from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import json, os
from rapidfuzz import process, fuzz
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = "agrobot_secret_key_2025"

DATA_FILE = "disease_data.json"
MODEL_PATH = "models/disease_model.h5"
SUPPORTED_LANGS = ["en", "hi", "bn", "ta", "te", "pa"]

# Load model at startup
model = load_model(MODEL_PATH)

# Class labels for predictions
classes = [
    "Pepper__bell__Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato__Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

# ----------------- Helpers -----------------
def load_data():
    if not os.path.exists(DATA_FILE):
        return {"symptoms": {}}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def detect_language(text: str) -> str:
    if not text:
        return "en"
    if any("\u0900" <= c <= "\u097F" for c in text):
        return "hi"
    if any("\u0980" <= c <= "\u09FF" for c in text):
        return "bn"
    if any("\u0B80" <= c <= "\u0BFF" for c in text):
        return "ta"
    if any("\u0C00" <= c <= "\u0C7F" for c in text):
        return "te"
    if any("\u0A00" <= c <= "\u0A7F" for c in text):
        return "pa"
    return "en"

def build_variant_map(data):
    variants = []
    for key, val in data.get("symptoms", {}).items():
        variants.append((key.lower(), key))
        translations = val.get("translations", {})
        if isinstance(translations, dict):
            for lang, text in translations.items():
                if text:
                    variants.append((text.lower(), key))
        elif isinstance(translations, list):
            for text in translations:
                if text:
                    variants.append((text.lower(), key))
    return variants

def match_symptom(user_text, data, threshold=65):
    variants = build_variant_map(data)
    choices = [v[0] for v in variants]
    if not choices:
        return None, 0
    match = process.extractOne(user_text.lower(), choices, scorer=fuzz.WRatio)
    if not match:
        return None, 0
    matched_text, score, idx = match
    if score < threshold:
        return None, score
    canonical = variants[idx][1]
    return canonical, score

def get_treatment_for_language(entry, lang):
    if not entry:
        return None
    treatments = entry.get("treatments", {})
    if isinstance(treatments, dict):
        return treatments.get(lang) or treatments.get("en") or "Treatment not available."
    if isinstance(treatments, str):
        return treatments
    return "Treatment not available."

def predict_disease_from_image(img_path):
    input_shape = model.input_shape[1:3]
    img = image.load_img(img_path, target_size=input_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_class = classes[np.argmax(predictions)]
    disease_name = predicted_class
    #data=load_data()
    #treatment_info = data["symptoms"].get(disease_name, {}).get("treatments", {})
    confidence = round(100 * np.max(predictions), 2)
    return predicted_class, confidence

# ----------------- Dummy users -----------------
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "farmer": {"password": "farmer123", "role": "farmer"}
}

# ----------------- Routes -----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = USERS.get(username)
        if user and user["password"] == password:
            session["username"] = username
            session["role"] = user["role"]
            flash(f"Welcome, {username}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password", "danger")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    role = session.get("role")
    return render_template("farmer.html" if role == "farmer" else "admin.html", role=role)

# ---------- Farmer chat endpoint ----------
@app.route("/ask", methods=["POST"])
def ask():
    if "username" not in session:
        return redirect(url_for("login"))
    data = load_data()
    user_text = request.form.get("symptom", "").strip()
    img_file = request.files.get("image")
    response = {"found": False, "score": 0, "disease": None, "treatment": None}

    # Case 1: If image uploaded → predict using CNN
    if img_file and img_file.filename != "":
        img_path = os.path.join("static", "uploads", img_file.filename)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img_file.save(img_path)
        disease_name, confidence = predict_disease_from_image(img_path)
        response["found"] = True
        response["disease"] = disease_name
        response["confidence"] = confidence
        entry = data.get("symptoms", {}).get(disease_name, {})
        treatment = get_treatment_for_language(entry, "en") if entry else "Treatment info not found in database."
        response["treatment"] = treatment
        flash(f"Disease: {disease_name} ({confidence}% confidence)", "info")
        flash(f"Suggested Treatment: {treatment}", "success")
        return jsonify(response) if request.is_json else redirect(url_for("dashboard"))

    # Case 2: If symptom text entered → use text-based matching
    lang = detect_language(user_text)
    if user_text:
        canonical, score = match_symptom(user_text, data, threshold=60)
        if canonical:
            entry = data["symptoms"].get(canonical, {})
            disease_name = entry.get("disease", canonical)
            treatment_text = get_treatment_for_language(entry, lang)
            response.update({
                "found": True,
                "score": score,
                "disease": disease_name,
                "treatment": treatment_text
            })
            flash(f"{disease_name}", "info")
            flash(treatment_text, "success")
        else:
            flash("Sorry, I could not identify the disease. Try rephrasing.", "warning")
    else:
        flash("Please enter symptoms or upload an image.", "warning")
    return jsonify(response) if request.is_json else redirect(url_for("dashboard"))

# ---------- Admin endpoints ----------
@app.route("/admin/add", methods=["POST"])
def admin_add():
    if "username" not in session or session.get("role") != "admin":
        return redirect(url_for("login"))
    data = load_data()
    sym = request.form.get("symptom").strip()
    disease = request.form.get("disease").strip()
    treatments = {}
    for lg in SUPPORTED_LANGS:
        val = request.form.get(f"t_{lg}", "").strip()
        if val:
            treatments[lg] = val
    translations = {}
    for lg in SUPPORTED_LANGS:
        val = request.form.get(f"s_{lg}", "").strip()
        if val:
            translations[lg] = val
    data.setdefault("symptoms", {})[sym] = {
        "disease": disease or sym,
        "translations": translations,
        "treatments": treatments
    }
    save_data(data)
    flash(f"Saved symptom '{sym}'", "success")
    return redirect(url_for("dashboard"))

@app.route("/admin/delete", methods=["POST"])
def admin_delete():
    if "username" not in session or session.get("role") != "admin":
        return redirect(url_for("login"))
    data = load_data()
    sym = request.form.get("symptom_key")
    if sym and sym in data.get("symptoms", {}):
        data["symptoms"].pop(sym, None)
        save_data(data)
        flash(f"Deleted '{sym}'", "success")
    else:
        flash("Symptom not found", "danger")
    return redirect(url_for("dashboard"))

@app.context_processor
def inject_data():
    data = load_data()
    return {"db_symptoms": data.get("symptoms", {})}

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for("login"))

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
