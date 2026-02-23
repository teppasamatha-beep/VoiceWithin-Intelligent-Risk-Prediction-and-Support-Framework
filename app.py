# ==========================================================
# Combined Flask App: QA + Mood Journal + Chatbot + Notifications
# (Merged with Ollama chatbot – NO OpenAI)
# ==========================================================

from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, send_file, session
)
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import gridfs
import io
import pickle
import numpy as np
import random, os, json
from collections import defaultdict, Counter
from dotenv import load_dotenv
import requests
from pywebpush import webpush
from apscheduler.schedulers.background import BackgroundScheduler

# ==========================================================
# Config & Setup
# ==========================================================
load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"  # change in production

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://Samatha:Samatha26@cluster0.x7razut.mongodb.net/?appName=Cluster0"
)
client = MongoClient(MONGO_URI)
db = client["SSP"]

# Collections
entries_col = db["Mood Journal"]
users_col = db["users"]
qa_col = db["QA"]
chats = db["Chatbot"]
fs = gridfs.GridFS(db, collection="images")

# ==========================================================
# VAPID / Notifications
# ==========================================================
VAPID_PUBLIC_KEY = "BK5IfpSj2hRtPdW13Q_66kLtypJSDpValC5LoED7ylls4ECXz9reQm9CIyp35uc2kCbnZImnQtv0eU9oYCMDll8"
VAPID_PRIVATE_KEY = "g9NKsBaz2wbb313JhSShqQg5sbZMmbMyd_K2GEt483c"
VAPID_CLAIMS = {"sub": "mailto:youremail@example.com"}

MESSAGES = [
    "🌞 Start your day strong! Every small step counts.",
    "💬 How are you feeling today? Take a moment to reflect.",
    "🌻 Remember, it’s okay to slow down and breathe.",
    "🌙 End your day with gratitude — you did your best.",
    "💡 Keep going, your story matters!",
    "🌼 You’re stronger than you think. Believe in yourself.",
    "🌈 Even tough days teach valuable lessons.",
]

# ==========================================================
# Load ML Model
# ==========================================================
with open("suicide_model(3).pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl(3)", "rb") as f:
    label_encoders = pickle.load(f)

# ==========================================================
# Helpers
# ==========================================================
def require_login_redirect():
    if "email" not in session:
        return redirect(url_for("login"))
    return None

def doc_to_public(doc):
    return {
        "_id": str(doc.get("_id")),
        "email": doc.get("email"),
        "title": doc.get("title", ""),
        "content": doc.get("content", ""),
        "mood": doc.get("mood", "neutral"),
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
        "images": [f"/image/{iid}" for iid in doc.get("image_ids", []) if iid]
    }

# ==========================================================
# Ollama Chatbot (from app.py 1)
# ==========================================================
def ollama_reply(user_message):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": f"""
You are VoiceWithin, a calm and empathetic therapist chatbot.
Keep replies short (3–5 sentences).
Be supportive and human.

User: {user_message}
VoiceWithin:
""",
                "stream": False,
                "options": {"num_predict": 120}
            },
            timeout=120
        )

        bot_reply = response.json().get("response", "").strip()
        return bot_reply or "I’m here with you 💙"

    except Exception as e:
        print("❌ Ollama error:", e)
        return "Sorry, I’m having trouble responding right now 😔"

# ==========================================================
# Auth / Navigation (UNCHANGED)
# ==========================================================
@app.route("/")
def intro():
    return render_template("intro.html")

@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if password != confirm_password:
            return jsonify({"status": "error", "message": "Passwords do not match"})

        if users_col.find_one({"email": email}):
            return jsonify({"status": "error", "message": "Email already registered"})

        users_col.insert_one({"name": name, "email": email, "password": password})
        return jsonify({"status": "success", "email": email})

    return render_template("index.html", public_key=VAPID_PUBLIC_KEY)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").lower()
        password = request.form.get("password", "")
        user = users_col.find_one({"email": email, "password": password})

        if not user:
            return "Invalid credentials", 401

        session["email"] = user["email"]
        session["name"] = user.get("name", "")
        return redirect(url_for("main_menu"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("intro"))

@app.route("/main_menu")
def main_menu():
    r = require_login_redirect()
    if r: return r
    return render_template("main_menu.html", name=session.get("name"))
@app.route("/survey")
def survey():
    r = require_login_redirect()
    if r: return r
    return render_template("survey.html")

@app.route("/QA")
def QA():
    r = require_login_redirect()
    if r: return r
    return render_template("QA.html")

@app.route("/predict", methods=["POST"])
def predict():
    r = require_login_redirect()
    if r: return r

    data = request.get_json(force=True)
    email = session["email"]

    feature_order = [
        "Choose your gender", "Age", "What is your course?", "Your current year",
        "Do you have Depression?", "Do you have Anxiety?", "Do you have Panic attack?",
        "Did you seek any specialist for a treatment?",
        "Have you ever experienced thoughts of ending your life (suicidal ideation)?",
        "Have you ever attempted to end your life in the past?",
        "Have you ever intentionally harmed yourself (e.g., cutting, burning, etc.)?",
        "Do you currently feel hopeless about your future?",
        "Have you ever experienced bullying (in person or online) that negatively affected your mental health?",
        "Are you currently experiencing serious family-related issues (e.g., conflict, abuse, divorce, financial stress)?",
        "Have you used substances (e.g., alcohol, tobacco, recreational drugs) to cope with stress or emotions?"
    ]

    input_data = []
    for feature in feature_order:
        val = data.get(feature, "")
        if feature in label_encoders:
            le = label_encoders[feature]
            try:
                transformed = le.transform([val])[0]
            except Exception:
                transformed = le.transform([le.classes_[0]])[0]
            input_data.append(transformed)
        else:
            try:
                input_data.append(float(val))
            except Exception:
                input_data.append(0.0)

    arr = np.array(input_data).reshape(1, -1)
    pred = model.predict(arr)[0]

    if "Suicide Rate" in label_encoders:
        target_le = label_encoders["Suicide Rate"]
        try:
            pred_text = target_le.inverse_transform([pred])[0]
        except Exception:
            pred_text = str(pred)
    else:
        pred_text = str(pred)

    suggestions = {
       "Low": "You’re doing awesome! 🚀 Keep shining, stay hydrated, and don’t forget to laugh 😄.",
            "Medium": "It seems like you’re going through some ups and downs 🌤️. Talking it out with a counselor or friend could help.",
            "High": "Things might feel heavy 💔. Please don’t face it alone — talking to a trusted person or professional can make a huge difference 🌱."
        }

    suggestion = suggestions.get(pred_text, "Please seek help if you feel unwell.")

    qa_doc = {
        "email": email,
        "timestamp": datetime.utcnow(),
        "answers": data,
        "risk_level": pred_text,
        "suggestion": suggestion
    }
    qa_col.insert_one(qa_doc)

    # Support older pattern where QA stored as a single doc with qa_history array:
    qa_col.update_one(
        {"email": email, "qa_history": {"$exists": True}},
        {"$push": {"qa_history": {"timestamp": datetime.utcnow(), "risk_level": pred_text, "answers": data, "suggestion": suggestion}}},
        upsert=False
    )

    session["prediction"] = pred_text
    session["suggestion"] = suggestion

    return jsonify({"redirect": url_for("result"), "prediction": pred_text, "suggestion": suggestion})

@app.route("/result")
def result():
    r = require_login_redirect()
    if r: return r
    prediction = session.get("prediction", "Not Available")
    suggestion = session.get("suggestion", "No suggestion available.")
    return render_template("result.html", prediction=prediction, suggestion=suggestion)

# ==========================================================
# Mood Journal Routes (unchanged)
# ==========================================================
@app.route("/journal")
def journal():
    r = require_login_redirect()
    if r: return r
    return render_template("new_entry.html")

@app.route("/add", methods=["POST"])
def add_entry():
    r = require_login_redirect()
    if r: return r

    title = request.form.get("title", "").strip() or "Untitled"
    content = request.form.get("content", "").strip()
    mood = request.form.get("mood", "neutral")
    created_at = datetime.utcnow()
    email = session.get("email")

    image_ids = []
    for file in request.files.getlist("image"):
        if file and file.filename:
            fid = fs.put(file.read(), filename=file.filename, content_type=file.mimetype)
            image_ids.append(str(fid))

    entries_col.insert_one({
        "email": email,
        "title": title,
        "content": content,
        "mood": mood,
        "image_ids": image_ids,
        "created_at": created_at
    })
    return redirect(url_for("history"))

@app.route("/entries")
def get_entries():
    if "email" not in session:
        return jsonify([])
    docs = entries_col.find({"email": session["email"]}).sort("created_at", -1)
    return jsonify([doc_to_public(d) for d in docs])

@app.route("/history")
def history():
    r = require_login_redirect()
    if r: return r
    docs = entries_col.find({"email": session["email"]}).sort("created_at", -1)
    entries = [doc_to_public(d) for d in docs]
    return render_template("hist.html", entries=entries)

@app.route("/edit/<id>", methods=["POST"])
def edit_entry(id):
    r = require_login_redirect()
    if r: return r
    oid = ObjectId(id)
    doc = entries_col.find_one({"_id": oid, "email": session["email"]})
    if not doc:
        return "Not found", 404

    title = request.form.get("title", "").strip() or "Untitled"
    content = request.form.get("content", "").strip()
    mood = request.form.get("mood", "neutral")

    files = request.files.getlist("image")
    if files and any(f.filename for f in files):
        for iid in doc.get("image_ids", []):
            try:
                fs.delete(ObjectId(iid))
            except:
                pass
        new_ids = []
        for f in files:
            fid = fs.put(f.read(), filename=f.filename, content_type=f.mimetype)
            new_ids.append(str(fid))
        entries_col.update_one({"_id": oid}, {"$set": {"title": title, "content": content, "mood": mood, "image_ids": new_ids}})
    else:
        entries_col.update_one({"_id": oid}, {"$set": {"title": title, "content": content, "mood": mood}})

    return redirect(url_for("history"))

@app.route("/delete/<id>", methods=["GET", "POST"])
def delete_entry(id):
    print("🧾 DELETE route triggered with ID:", id)
    try:
        oid = ObjectId(str(id))
    except Exception as e:
        print("❌ Invalid ObjectId:", e)
        return jsonify({"error": "Invalid ID"}), 400

    doc = entries_col.find_one({"_id": oid})
    if not doc:
        print("❌ No document found for this ID")
        return jsonify({"error": "Not found"}), 404

    for iid in doc.get("image_ids", []):
        try:
            fs.delete(ObjectId(iid))
        except Exception as e:
            print("⚠️ Error deleting image:", e)

    result = entries_col.delete_one({"_id": oid})
    print("🗑️ Deleted count:", result.deleted_count)

    return jsonify({"success": True})

@app.route("/image/<image_id>")
def serve_image(image_id):
    try:
        grid_out = fs.get(ObjectId(image_id))
        return send_file(io.BytesIO(grid_out.read()), mimetype=grid_out.content_type)
    except:
        return "Image not found", 404

@app.route("/mood_stats")
def mood_stats():
    if "email" not in session:
        return jsonify({"daily_summary": [], "monthly_summary": []})
    docs = list(entries_col.find({"email": session["email"]}))
    daily = defaultdict(list)
    monthly = defaultdict(list)
    for d in docs:
        if not d.get("created_at"): continue
        day = d["created_at"].strftime("%Y-%m-%d")
        month = d["created_at"].strftime("%Y-%m")
        mood = d.get("mood", "neutral")
        daily[day].append(mood)
        monthly[month].append(mood)
    daily_summary = [{"date": k, "mood": Counter(v).most_common(1)[0][0]} for k, v in sorted(daily.items())]
    monthly_summary = [{"month": k, "mood": Counter(v).most_common(1)[0][0]} for k, v in sorted(monthly.items())]
    return jsonify({"daily_summary": daily_summary, "monthly_summary": monthly_summary})

@app.route("/improvement")
def improvement():
    if "email" not in session:
        return redirect(url_for("login"))

    email = session["email"]

    try:
        qa_data = []
        qa_docs = list(qa_col.find({"email": email}))

        # ✅ Handle both styles of QA data
        for doc in qa_docs:
            if "qa_history" in doc:
                for entry in doc["qa_history"]:
                    ts = entry.get("timestamp")
                    risk = entry.get("risk_level", "Unknown")
                    if ts and risk:
                        qa_data.append({
                            "date": ts.strftime("%Y-%m-%d"),
                            "risk": risk
                        })
            else:
                ts = doc.get("timestamp")
                risk = doc.get("risk_level", "Unknown")
                if ts and risk:
                    qa_data.append({
                        "date": ts.strftime("%Y-%m-%d"),
                        "risk": risk
                    })

        qa_data.sort(key=lambda x: x["date"])

        # ✅ Get mood journal entries
        mood_entries = list(entries_col.find({"email": email}))
        mood_data = []
        for entry in mood_entries:
            created_at = entry.get("created_at")
            if created_at:
                mood_data.append({
                    "date": created_at.strftime("%Y-%m-%d"),
                    "mood": entry.get("mood", "")
                })

        last_risk = qa_data[-1]["risk"] if qa_data else "N/A"
        total_journal_entries = len(mood_data)

        return render_template(
            "improve.html",
            risk_data=qa_data,
            mood_data=mood_data,
            last_risk=last_risk,
            total_journal_entries=total_journal_entries
        )

    except Exception as e:
        print("Error in /improvement:", e)
        return render_template(
            "improve.html",
            message="Error loading improvement data.",
            risk_data=[],
            mood_data=[],
            last_risk="N/A",
            total_journal_entries=0
        )

# ==========================================================
# Chatbot Routes (PRIVACY SAFE)
# ==========================================================
@app.route("/chatbot")
def chatbot_home():
    r = require_login_redirect()
    if r: return r
    return render_template(
        "chatbot.html",
        email=session["email"],
        username=session.get("name", "")
    )
def detect_sentiment(text):
    text = text.lower()

    positive = ["happy", "good", "great", "excited", "relaxed", "peaceful"]
    negative = ["sad", "depressed", "angry", "lonely", "tired", "anxious", "stress"]
    
    if any(word in text for word in positive):
        return "positive"
    if any(word in text for word in negative):
        return "negative"
    
    return "neutral"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if "email" not in session:
            return jsonify({"reply": "Please login first"}), 401

        data = request.get_json()
        user_message = data.get("message", "").strip()
        chat_name = data.get("chat_name", "").strip()
        email = session["email"]

        if not user_message:
            return jsonify({"reply": "Please type something 💬"})

        if not chat_name:
            chat_name = user_message[:40]

        # ✅ SENTIMENT OF USER MESSAGE (CORRECT PLACE)
        user_sentiment = detect_sentiment(user_message)

        # 🤖 Bot reply (NO sentiment here)
        bot_reply = ollama_reply(user_message)

        # 🧠 Save USER message WITH sentiment
        chats.update_one(
            {"email": email, "chat_name": chat_name},
            {"$push": {"messages": {
                "sender": "user",
                "text": user_message,
                "sentiment": user_sentiment,
                "timestamp": datetime.utcnow().isoformat()
            }}},
            upsert=True
        )

        # 🤖 Save BOT reply (NO sentiment)
        chats.update_one(
            {"email": email, "chat_name": chat_name},
            {"$push": {"messages": {
                "sender": "bot",
                "text": bot_reply,
                "timestamp": datetime.utcnow().isoformat()
            }}}
        )

        return jsonify({
            "reply": bot_reply,
            "chat_name": chat_name,
            "sentiment": user_sentiment   # optional for frontend
        })

    except Exception as e:
        print("⚠️ Chat error:", e)
        return jsonify({"reply": "Something went wrong 😔"}), 500


@app.route("/get_chats")
def get_chats():
    if "email" not in session:
        return jsonify({})

    email = session["email"]
    docs = chats.find({"email": email})
    result = {}

    for doc in docs:
        result[doc["chat_name"]] = doc.get("messages", [])

    return jsonify(result)

# ==========================================================
# Notifications
# ==========================================================
@app.route("/save_subscription", methods=["POST"])
def save_subscription():
    data = request.get_json()
    email = data.get("email")
    sub = data.get("subscription")

    if not email or not sub:
        return jsonify({"error": "Missing email or subscription"}), 400

    db.subscriptions.update_one(
        {"email": email},
        {"$set": {"subscription": sub}},
        upsert=True
    )
    print(f"📬 Saved push subscription for {email}")
    return jsonify({"status": "success"})

def send_push_notification():
    message = random.choice(MESSAGES)
    data = {"title": "VoiceWithin 💬", "body": message}
    for sub_doc in db.subscriptions.find():
        sub = sub_doc.get("subscription")
        if not sub:
            continue
        try:
            webpush(
                subscription_info=sub,
                data=json.dumps(data),
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims=VAPID_CLAIMS
            )
        except Exception as e:
            print("❌ Push failed for a subscription:", e)
    print(f"✅ Notification sent at {datetime.utcnow()}: {message}")

@app.route("/test")
def test_push():
    send_push_notification()
    return "✅ Test notification sent"
# Schedule notifications (10:00 and 19:00 server local time)
scheduler = BackgroundScheduler()
scheduler.add_job(send_push_notification, "cron", hour=10, minute=0)
scheduler.add_job(send_push_notification, "cron", hour=19, minute=0)
scheduler.start()
# ==========================================================
# Health Check
# ==========================================================
@app.route("/ping")
def ping():
    return "✅ Flask is running and ready!"

# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
