from fastapi import FastAPI, Request
import joblib
from pathlib import Path
import os
import random
import httpx

app = FastAPI()

# ---------------- HEALTH CHECK (REQUIRED FOR RENDER) ---------------- #
@app.get("/")
def health_check():
    return {"status": "ok"}

# ---------------- LOAD ML MODEL ---------------- #
MODEL_PATH = Path("NoteBook/intent_model.pkl")

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# ---------------- TELEGRAM CONFIG ---------------- #
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
        "I see this is a billing-related issue.\n\n"
        "Please tell me:\n"
        "💳 Are you unable to make a payment?\n"
        "📄 Is there an incorrect charge?\n"
        "🧾 Are you missing a bill or receipt?"
    ),
    (
        "To avoid payment issues:\n\n"
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
        "📅 When did you contact support?\n"
        "📞 Was it a call, chat, or email?"
    ),
    (
        "Thank you for sharing this.\n"
        "If the issue is still unresolved, I can guide you to the right escalation option."
    )
]

ACCOUNT_RESPONSES = [
    (
        "I see there’s an issue related to your account.\n\n"
        "1️⃣ Unable to access your account?\n"
        "2️⃣ Verification or missing details?\n"
        "3️⃣ Billing or profile update issue?"
    ),
    (
        "To proceed:\n\n"
        "🔹 Ensure your registered email and phone are active\n"
        "🔹 Check if you recently changed devices\n\n"
        "Let me know what applies."
    )
]

CONNECTIVITY_RESPONSES = [
    (
        "I understand you’re facing a connectivity issue.\n\n"
        "1️⃣ Wi-Fi or mobile data?\n"
        "2️⃣ Other devices affected?\n"
        "3️⃣ Since when?"
    ),
    (
        "Please try these quick checks:\n\n"
        "🔹 Restart router/modem\n"
        "🔹 Check cables\n"
        "🔹 Toggle airplane mode\n\n"
        "Tell me the result."
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

# ---------------- TELEGRAM WEBHOOK ---------------- #
@app.post("/telegram-webhook")
async def telegram_webhook(req: Request):
    data = await req.json()

    if "message" not in data or "text" not in data["message"]:
        return {"status": "ignored"}

    chat_id = data["message"]["chat"]["id"]
    user_text = data["message"]["text"].strip().lower()

    # ---- BASIC COMMANDS ---- #
    if user_text in ["hello", "hi", "hey"]:
        reply_text = "Hello 👋 How can I help you today?"

    elif user_text in ["thanks", "thank you", "thx"]:
        reply_text = "You’re welcome 😊 Happy to help!"

    elif user_text in ["bye", "goodbye"]:
        reply_text = "Goodbye 👋 Take care!"

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

    # ---- ML INTENT HANDLING ---- #
    else:
        intent = model.predict([user_text])[0]
        confidence = model.predict_proba([user_text]).max()

        if confidence < 0.6:
            reply_text = "I’m not fully sure I understood. Could you please explain again?"
        else:
            reply_text = random.choice(
                INTENT_RESPONSE_MAP.get(intent, GENERAL_QUERY_RESPONSES)
            )

    # ---- SEND MESSAGE (ASYNC SAFE) ---- #
    async with httpx.AsyncClient() as client:
        await client.post(
            TELEGRAM_API,
            json={"chat_id": chat_id, "text": reply_text}
        )

    return {"status": "ok"}
