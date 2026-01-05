from fastapi import FastAPI, Request
import joblib
from pathlib import Path
import requests
import os
import random

app = FastAPI()

# Load ML pipeline model
MODEL_PATH = Path("NoteBook") / "intent_model.pkl"
model = joblib.load(MODEL_PATH)

# Telegram config
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable not set")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


# ---------------- INTENT RESPONSES ---------------- #

GENERAL_QUERY_RESPONSES = [
    "I want to make sure I understand correctly. Could you please explain your issue in a bit more detail?",
    "Can you tell me what exactly you need help with right now?",
    "I’m here to help — is this about your account, billing, or connectivity?"
]

POSITIVE_FEEDBACK_RESPONSES = [
    "That’s great to hear! 😊 I’m glad someone was able to help you.",
    "Thanks for sharing your feedback! If you need anything else, I’m here.",
    "Happy to know your issue was resolved. Have a great day!"
]

BILLING_RESPONSES = [
    (
        "I see this is a billing-related issue. Let’s check it carefully.\n\n"
        "Please tell me:\n"
        "💳 Are you unable to make a payment?\n"
        "📄 Is there an incorrect charge?\n"
        "🧾 Are you missing a bill or receipt?"
    ),
    (
        "To avoid any payment issues:\n\n"
        "🔹 Check if your payment method is active\n"
        "🔹 Confirm billing address is correct\n"
        "🔹 Try again after some time\n\n"
        "Let me know what error you’re seeing."
    )
]

CUSTOMER_SERVICE_RESPONSES = [
    (
        "I’m really sorry about the experience you had with customer support.\n\n"
        "Please tell me what went wrong:\n"
        "🔹 Long wait times?\n"
        "🔹 Incorrect information?\n"
        "🔹 No response at all?"
    ),
    (
        "That shouldn’t have happened, and I understand your frustration.\n\n"
        "To help you better:\n"
        "📅 When did you contact support?\n"
        "📞 Was it call, chat, or email?"
    ),
    (
        "Thank you for sharing this. I’ll make sure this gets addressed.\n\n"
        "If the issue is still unresolved, I can guide you to the right escalation option."
    )
]

ACCOUNT_RESPONSES = [
    (
        "I see there’s an issue related to your account. I’ll help you sort this out.\n\n"
        "Please tell me:\n"
        "1️⃣ Are you unable to access your account?\n"
        "2️⃣ Is this related to verification or missing details?\n"
        "3️⃣ Are you trying to pay a bill or update information?"
    ),
    (
        "Thanks for the details. To move forward:\n\n"
        "🔹 Make sure your registered phone number and email are active\n"
        "🔹 Check if you recently changed devices or address\n\n"
        "Let me know which of these applies to you."
    ),
    (
        "I understand this can be annoying. If the system is not finding your account:\n\n"
        "📄 Do you have any bill, receipt, or device serial number?\n"
        "🏬 Did you sign up online or at a store?"
    )
]

CONNECTIVITY_RESPONSES = [
    (
        "I understand you’re facing a connectivity issue. Let’s fix this step by step.\n\n"
        "First, please tell me:\n"
        "1️⃣ Is this Wi-Fi or mobile data?\n"
        "2️⃣ Are other devices also affected?\n"
        "3️⃣ Since when is the issue happening?"
    ),
    (
        "Thanks for reporting this. Please try these quick checks:\n\n"
        "🔹 Restart your router or modem\n"
        "🔹 Check if cables are properly connected\n"
        "🔹 Turn airplane mode ON and OFF\n\n"
        "Let me know what you observe after this."
    ),
    (
        "I know internet issues are frustrating. If the problem is still there:\n\n"
        "📍 Please confirm your location (city/area)\n"
        "📶 Check if signal bars are low\n"
        "🕒 Tell me if this happens at specific times"
    )
]

INTENT_RESPONSE_MAP = {
    "connectivity_issue": CONNECTIVITY_RESPONSES,
    "account_issue": ACCOUNT_RESPONSES,
    "customer_service_complaint": CUSTOMER_SERVICE_RESPONSES,
    "billing_issue": BILLING_RESPONSES,
    "positive_feedback": POSITIVE_FEEDBACK_RESPONSES,
    "general_query": GENERAL_QUERY_RESPONSES
}


@app.post("/telegram-webhook")
async def telegram_webhook(req: Request):
    data = await req.json()

    if "message" not in data or "text" not in data["message"]:
        return {"status": "ignored"}

    chat_id = data["message"]["chat"]["id"]
    user_text = data["message"]["text"].strip().lower()

    # ---------- BASIC GREETINGS / POLITENESS ---------- #
    if user_text in ["hello", "hi", "hey"]:
        reply_text = "Hello 👋 How can I help you today?"

    elif user_text in ["thanks", "thank you", "thx"]:
        reply_text = "You’re welcome 😊 Happy to help!"

    elif user_text in ["bye", "goodbye", "see you"]:
        reply_text = "Goodbye 👋 Take care!"

    # ---------- COMMANDS ---------- #
    elif user_text == "/start":
        reply_text = "Hello 👋 I am an AI support bot. How can I help you today?"

    elif user_text == "/help":
        reply_text = (
            "You can ask me about:\n"
            "- Account issues\n"
            "- Connectivity problems\n"
            "- Billing issues\n"
            "- Customer service complaints"
        )
        
    elif user_text in [
    "yes solved", "solved", "done", "fixed", "issue resolved",
    "problem solved", "now its working", "now it's working",
    "working now", "resolved", "its working"
    ]:
        reply_text = (
            "That’s great to hear! 😊\n"
            "I’m glad the issue is resolved.\n\n"
            "If you need help with anything else, just let me know!"
        )

    # ---------- ML INTENT HANDLING ---------- #
    else:
        intent = model.predict([user_text])[0]
        confidence = model.predict_proba([user_text]).max()

        if confidence < 0.6:
            reply_text = "I’m not fully sure I understood. Could you please explain again?"
        else:
            responses = INTENT_RESPONSE_MAP.get(intent, GENERAL_QUERY_RESPONSES)
            reply_text = random.choice(responses)

    # ---------- SEND MESSAGE ---------- #
    requests.post(
        TELEGRAM_API,
        json={
            "chat_id": chat_id,
            "text": reply_text
        }
    )

    return {"status": "ok"}
