import os
import json
import re
from io import BytesIO
from email import message_from_bytes
from email.policy import default as default_policy
from dotenv import load_dotenv
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq

load_dotenv()

class ClassifierAgent:
    def __init__(self, temperature: float = 0.0):
        groq_key = os.getenv("GROQ_API_KEY")
        groq_model = os.getenv("GROQ_MODEL")
        if not groq_key or not groq_model:
            raise ValueError("GROQ_API_KEY and GROQ_MODEL must be set in .env")

        self.llm = ChatGroq(api_key=groq_key, model=groq_model, temperature=temperature)
        self.max_snippet_chars = 4096  # Increased for more context

        # Semantic keyword clusters with weights
        self.intent_keywords = {
            "Invoice": [
                (r"\b(tax\s*invoice|invoice|bill\s*of\s*supply|cash\s*memo)\b", 2),  # High weight for explicit terms
                (r"\b(total|subtotal|due\s*date|payment|order\s*number|amount|invoice\s*number)\b", 1),
                (r"\b(\$|₹)?\s*\d+[.,]\d+\b", 1),  # Monetary values
            ],
            "RFQ": [
                (r"\b(quote|request|quantity|product|pricing|specification)\b", 1),
                (r"\b(inquiry|procurement|order\s*request)\b", 1),
            ],
            "Complaint": [
                (r"\b(complaint|defective|issue|problem|replace|refund)\b", 1),
                (r"\b(customer|support|service)\b", 1),
            ],
            "Regulation": [
                (r"\b(section|act|regulation|compliance|policy|law)\b", 1),
                (r"\b(legal|authority|standard)\b", 1),
            ],
            "Fraud Risk": [
                (r"\b(fraud|suspicious|mismatch|investigate|discrepancy)\b", 1),
                (r"\b(risk|alert|warning)\b", 1),
            ],
            "Unknown": [],
        }

        self.intent_patterns = {
            intent: [(re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in patterns]
            for intent, patterns in self.intent_keywords.items()
        }

        # Few-shot examples with semantic context
        few_shot_examples = [
            {
                "format": "Email",
                "text": "Subject: Urgent complaint #12987\nThe product I received is defective and doesn’t work. Please replace it immediately.",
                "intent": "Complaint",
                "context": "Mentions 'defective' and 'replace' in a customer service context."
            },
            {
                "format": "JSON",
                "text": "{\"product\": \"Widget A\", \"quantity\": 100, \"request\": \"pricing details\"}",
                "intent": "RFQ",
                "context": "Contains procurement terms like 'quantity' and 'request' for pricing."
            },
            {
                "format": "PDF",
                "text": "Tax Invoice/Bill of Supply Order Number: 407-0648430-7690762 Invoice Number: DED4-3622 Total: ₹1,699.00 Seller: CLICKTECH RETAIL PRIVATE LIMITED",
                "intent": "Invoice",
                "context": "Contains 'Tax Invoice,' 'Invoice Number,' 'Total,' and monetary value in a billing context."
            },
            {
                "format": "PDF",
                "text": "INVOICE #INV-2025-0615 Date: June 15, 2025 Total Due: $16,344.25",
                "intent": "Invoice",
                "context": "Mentions 'INVOICE,' 'Total Due,' and monetary value in a financial context."
            },
            {
                "format": "PDF",
                "text": "Section 32 of the Data Privacy Act requires encrypted data storage.",
                "intent": "Regulation",
                "context": "Mentions 'Section' and 'Act' in a legal context."
            },
            {
                "format": "Email",
                "text": "Subject: Fraud Alert\nSuspected mismatch in tax IDs on invoice #4532. Please investigate.",
                "intent": "Fraud Risk",
                "context": "Mentions 'fraud' and 'investigate' in a risk-related context."
            },
            {
                "format": "PDF",
                "text": "Attention is all you need.",
                "intent": "Unknown",
                "context": "Lacks specific intent-related keywords or clear business context."
            },
        ]

        example_prompt = PromptTemplate(
            input_variables=["format", "text", "intent", "context"],
            template="Format: {{ format }}\nText: {{ text }}\nContext: {{ context }}\nIntent: {{ intent }}\n",
            template_format="jinja2"
        )

        prefix = (
            "You are a classifier tasked with determining the business intent of a file based on its format and content. "
            "Analyze the semantic context of keywords (e.g., financial terms like 'invoice' and 'total' for invoices, "
            "legal terms like 'act' for regulations) and choose exactly one intent from [RFQ, Complaint, Invoice, Regulation, Fraud Risk, Unknown]. "
            "Prioritize the meaning and role of keywords in the context, not just their presence.\n\n"
        )
        suffix = "Format: {{ input_format }}\nText: {{ input_text }}\nIntent:"

        self.few_shot_prompt = FewShotPromptTemplate(
            examples=few_shot_examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input_format", "input_text"],
            example_separator="\n---\n",
            template_format="jinja2"
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.few_shot_prompt)

    def process(self, raw_bytes: bytes, filename: str, metadata: dict = None) -> dict:
        fmt = self._format_from_filename(filename)
        snippet = self._bytes_to_text(raw_bytes)

        if len(snippet) > self.max_snippet_chars:
            snippet = snippet[:self.max_snippet_chars] + "..."

        # Check metadata for document_type
        if metadata and metadata.get("extraction", {}).get("document_type") == "Tax Invoice":
            return {"source": "classifier", "format": fmt, "intent": "Invoice"}

        # Semantic scoring
        intent_scores = self._score_intents(snippet, fmt)
        if fmt == "PDF" and intent_scores.get("Invoice", 0) >= 3:  # Require strong invoice evidence
            return {"source": "classifier", "format": fmt, "intent": "Invoice"}

        # Fallback to LLM
        llm_output = self.chain.run(input_format=fmt, input_text=snippet)
        intent = self._parse_intent(llm_output)
        return {"source": "classifier", "format": fmt, "intent": intent}

    def _format_from_filename(self, filename: str) -> str:
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext == "json":
            return "JSON"
        if ext == "pdf":
            return "PDF"
        if ext in {"eml", "txt", "email"}:
            return "Email"
        return "Unknown"

    def _bytes_to_text(self, raw_bytes: bytes) -> str:
        # Email
        try:
            msg = message_from_bytes(raw_bytes, policy=default_policy)
            headers = [f"Subject: {msg.get('Subject', '')}" if msg.get("Subject") else "",
                      f"From: {msg.get('From', '')}" if msg.get("From") else ""]
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain" and part.get_payload(decode=True):
                        body += part.get_payload(decode=True).decode("utf-8", errors="ignore")
            else:
                body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
            combined = " ".join(headers) + " " + body
            return re.sub(r"\s+", " ", combined).strip()
        except:
            pass

        # JSON
        try:
            text = raw_bytes.decode("utf-8", errors="ignore")
            data = json.loads(text)
            return json.dumps(data, indent=2)
        except:
            pass

        # PDF
        try:
            reader = PdfReader(BytesIO(raw_bytes))
            pages = []
            for i, page in enumerate(reader.pages):
                if i >= 2:
                    break
                txt = page.extract_text() or ""
                txt = re.sub(r"\s+", " ", txt)  # Normalize whitespace
                txt = re.sub(r"\|\s*\|", " ", txt)  # Remove table artifacts
                pages.append(txt)
            return " ".join(pages).strip()
        except:
            pass

        # UTF-8 fallback
        try:
            decoded = raw_bytes.decode("utf-8", errors="ignore")
            return re.sub(r"\s+", " ", decoded).strip()
        except:
            return repr(raw_bytes[:200])

    def _score_intents(self, snippet: str, fmt: str) -> dict:
        scores = {intent: 0 for intent in self.intent_keywords}
        for intent, patterns in self.intent_patterns.items():
            if intent == "Unknown":
                continue
            for pattern, weight in patterns:
                if pattern.search(snippet):
                    scores[intent] += weight
        return scores

    def _parse_intent(self, llm_output: str) -> str:
        match = re.search(r"Intent:\s*([A-Za-z ]+)", llm_output)
        return match.group(1).strip() if match else "Unknown"