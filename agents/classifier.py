# agents/classifier_agent.py

import os
import json
import re
from io import BytesIO
from datetime import datetime

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader

load_dotenv()  # looks for a .env file in your project root

class ClassifierAgent:
    """
    ClassifierAgent:
      - Initializes a Few‑Shot LLMChain to classify raw bytes into:
        format ∈ { "JSON", "Email", "PDF" }
        intent ∈ { "RFQ", "Complaint", "Invoice", "Regulation", "Fraud Risk" }
      - Exposes `process(raw_bytes)` → dict:
          { "source":"classifier", "format":…, "intent":… }
    """

    def __init__(self, temperature: float = 0.0, model_name: str = "gpt-4"):
        # 1) Pull API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
        # 2) Create ChatOpenAI client
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)

        # 3) Prepare few-shot examples and prompt template
        few_shot_examples = [
            {
                "text": "Subject: Urgent complaint about order #12987\nI received the wrong item...",
                "format": "Email",
                "intent": "Complaint"
            },
            {
                "text": "{\n  \"product\": \"Widget A\",\n  \"quantity\": 100,\n  \"delivery_date\": \"2024-09-01\"\n}",
                "format": "JSON",
                "intent": "RFQ"
            },
            {
                "text": "Invoice #: 34897\nTotal Due: $5,204.99\nDue Date: 2024-08-10",
                "format": "PDF",
                "intent": "Invoice"
            },
            {
                "text": "According to Section 32 of the Data Privacy Act...",
                "format": "PDF",
                "intent": "Regulation"
            },
            {
                "text": "We suspect this invoice is fraudulent due to mismatched tax IDs...",
                "format": "Email",
                "intent": "Fraud Risk"
            },
        ]

        example_prompt = PromptTemplate(
            input_variables=["text", "format", "intent"],
            template="Text: {text}\nFormat: {format}\nIntent: {intent}\n"
        )

        prefix = (
            "Classify the given text into Format and Intent.\n"
            "Choose Format from [JSON, Email, PDF].\n"
            "Choose Intent from [RFQ, Complaint, Invoice, Regulation, Fraud Risk].\n\n"
        )
        suffix = "Text: {input_text}\nFormat:"

        self.few_shot_prompt = FewShotPromptTemplate(
            examples=few_shot_examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input_text"]
        )

        # Create the LLMChain
        self.chain = LLMChain(llm=self.llm, prompt=self.few_shot_prompt)

    def process(self, raw_bytes: bytes, metadata: dict = None) -> dict:
        """
        1) Convert raw_bytes → text snippet
        2) Call the LLMChain to get back a response like:
            "Format: Email\nIntent: Complaint"
        3) Parse that out and return a dict:
            { "source":"classifier", "format":..., "intent":... }
        """
        text_snippet = self._bytes_to_text(raw_bytes)
        llm_output = self.chain.run(input_text=text_snippet)

        parsed = self._parse_llm_response(llm_output)
        return {
            "source": "classifier",
            "format": parsed.get("format", "Unknown"),
            "intent": parsed.get("intent", "Unknown")
        }

    def _bytes_to_text(self, raw_bytes: bytes) -> str:
        """
        Heuristic conversion of raw_bytes → text, so the LLM
        can see a human-readable snippet:
          1) Try JSON parse → pretty‑print
          2) Else try UTF‑8 / Latin‑1 decode (for emails and text)
          3) Else try PDF text extraction via PyPDF2
        """
        # Attempt 1: JSON?
        try:
            obj = json.loads(raw_bytes.decode("utf-8"))
            pretty = json.dumps(obj, indent=2)
            return pretty
        except Exception:
            pass

        # Attempt 2: Plaintext decode
        try:
            return raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            pass

        # Attempt 3: PDF text extraction
        try:
            reader = PdfReader(BytesIO(raw_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except Exception:
            pass

        # Fallback: raw base64 or repr
        return repr(raw_bytes[:500])  # show first 500 bytes in repr

    def _parse_llm_response(self, llm_output: str) -> dict:
        """
        Given an LLM output like:
            "Format: Email\nIntent: Complaint"
        extract the `format` and `intent` lines.
        """
        format_match = re.search(r"Format:\s*(\w+)", llm_output)
        intent_match = re.search(r"Intent:\s*([A-Za-z ]+)", llm_output)

        return {
            "format": format_match.group(1) if format_match else "Unknown",
            "intent": intent_match.group(1).strip() if intent_match else "Unknown"
        }
