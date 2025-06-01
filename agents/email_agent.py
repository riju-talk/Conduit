import os
import re
from uuid import uuid4
from email import message_from_bytes
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

class EmailAgent:
    """
    EmailAgent:
      - Parses raw email bytes (MIME .eml or plain text)
      - Extracts: sender, subject, body_summary, urgency, tone, thread_id
      - Suggests an action: "escalate" → CRM or "log" → database
      - Uses a small LLMChain to detect tone (angry/polite/threatening/spam)
    """

    def __init__(self, temperature: float = 0.0):
        # Instantiate ChatOpenAI for tone classification
        model_name = os.getenv("GROQ_MODEL")
        self.llm = ChatGroq(model_name=model_name, temperature=temperature)

        # Tone classification prompt
        tone_prompt = PromptTemplate(
            input_variables=["email_body"],
            template=(
                "Classify the tone of this email body into one of: [angry, polite, threatening, spam].\n\n"
                "Email Body:\n{email_body}\n\n"
                "Tone:"
            )
        )
        self.tone_chain = LLMChain(llm=self.llm, prompt=tone_prompt)

        # Define urgency keywords
        self.urgent_keywords = ["urgent", "asap", "immediately", "as soon as possible"]

    def process(self, raw_bytes: bytes, metadata: Dict[str, Any]) -> dict:
        """
        1) Parse MIME headers/body
        2) Extract: sender, subject, body
        3) Compute: urgency, tone, body_summary, thread_id
        4) Suggest action
        5) Return:
            {"source":"email_agent", "data":{...}, "action_suggestion":{...}}
        """
        # 1) Parse email
        msg = message_from_bytes(raw_bytes)
        sender = msg.get("From", "")
        subject = msg.get("Subject", "")
        in_reply_to = msg.get("In-Reply-To", None)

        # 2) Extract plain‑text body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain" and part.get_payload(decode=True):
                    body += part.get_payload(decode=True).decode("utf-8", errors="ignore")
        else:
            try:
                body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
            except Exception:
                body = str(msg.get_payload())

        # 3) Compute fields
        body_summary = self._summarize_body(body)
        urgency = self._get_urgency(subject, body)
        tone    = self._get_tone(body)
        thread_id = in_reply_to if in_reply_to else uuid4().hex

        data = {
            "sender":       sender,
            "subject":      subject,
            "body_summary": body_summary,
            "urgency":      urgency,
            "tone":         tone,
            "thread_id":    thread_id
        }

        # 4) Decide action
        if urgency == "high" or tone == "angry":
            action_suggestion = {"action": "escalate", "target": "crm"}
        else:
            action_suggestion = {"action": "log", "target": "database"}

        return {
            "source": "email_agent",
            "data": data,
            "action_suggestion": action_suggestion
        }

    def _summarize_body(self, body: str) -> str:
        """
        For simplicity, just return the first 200 characters.
        """
        return body.strip()[:200]

    def _get_urgency(self, subject: str, body: str) -> str:
        """
        If any urgent keyword appears in subject or body → "high", else "normal".
        """
        text = (subject + " " + body).lower()
        for kw in self.urgent_keywords:
            if kw in text:
                return "high"
        return "normal"

    def _get_tone(self, body: str) -> str:
        if not body.strip():
            return "polite"

        # Truncate to first 1000 chars to avoid very long prompts
        truncated = body[:1000]
        llm_response = self.tone_chain.run(email_body=truncated).strip().lower()

        # Post‑process to ensure one of the three labels
        for label in ["angry", "polite", "threatening", "spam"]:
            if label in llm_response:
                return label
        return "polite"
