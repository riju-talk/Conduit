# agents/pdf_agent.py

import os
import re
from io import BytesIO
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

class PDFAgent:
    """
    PDFAgent:
      - Extracts full text from PDF bytes
      - If metadata['intent'] == "Invoice":
          • Parse line items, compute invoice_total
          • If invoice_total > 10000 → flag_compliance→risk_alert
          • Else → archive→database
      - If metadata['intent'] == "Regulation":
          • Search for policy keywords (GDPR, FDA, etc.)
          • If any found → flag_compliance→risk_alert
          • Else → archive→database
      - Else (generic PDF) → return full_text, action → archive→database
    """

    def __init__(self):
        # Define a list of policy keywords to search for in regulations
        self.policy_keywords = ["GDPR", "FDA", "HIPAA", "PCI-DSS"]

    def process(self, raw_bytes: bytes, metadata: Dict[str, Any]) -> dict:
        """
        1) Extract `full_text` via PyPDF2
        2) Branch on metadata["intent"]
           - If "Invoice": parse invoice, compute invoice_total
           - If "Regulation": find policy keywords
           - Else: generic
        3) Build `data` dict + `action_suggestion`
        4) Return:
           { "source":"pdf_agent", "data":{…}, "action_suggestion":{…} }
        """
        intent = metadata.get("intent", "").lower()
        full_text = self._extract_text(raw_bytes)

        data: Dict[str, Any] = { "full_text": full_text }
        action_suggestion: Dict[str, str]

        if intent == "invoice":
            line_items, invoice_total = self._parse_invoice(full_text)
            data["line_items"] = line_items
            data["invoice_total"] = invoice_total

            if invoice_total > 10000:
                action_suggestion = {"action": "flag_compliance", "target": "risk_alert"}
            else:
                action_suggestion = {"action": "archive", "target": "database"}

        elif intent == "regulation":
            policy_mentions = self._find_policies(full_text)
            data["policy_mentions"] = policy_mentions

            if policy_mentions:
                action_suggestion = {"action": "flag_compliance", "target": "risk_alert"}
            else:
                action_suggestion = {"action": "archive", "target": "database"}

        else:
            # Generic PDF: just archive
            action_suggestion = {"action": "archive", "target": "database"}

        return {
            "source": "pdf_agent",
            "data": data,
            "action_suggestion": action_suggestion
        }

    def _extract_text(self, raw_bytes: bytes) -> str:
        """
        Use PyPDF2 to read bytes → full text (all pages).
        """
        reader = PdfReader(BytesIO(raw_bytes))
        pages_text = []
        for page in reader.pages:
            try:
                txt = page.extract_text()
            except Exception:
                txt = ""
            pages_text.append(txt or "")
        return "\n".join(pages_text)

    def _parse_invoice(self, text: str) -> Tuple[List[Dict[str, Any]], float]:
        """
        Attempt to parse line items via regex. 
        This is a heuristic: look for lines with "ItemName   qty   $unit   $line_total".
        Return (line_items, invoice_total).
        """
        line_items: List[Dict[str, Any]] = []
        invoice_total = 0.0

        # Simple regex that matches common invoice line patterns, e.g.:
        # WidgetA   5   $20.00   $100.00
        # Adjust this regex as needed for your PDF’s structure
        pattern = re.compile(
            r"(?P<item>[A-Za-z0-9 \-]+)\s+"
            r"(?P<qty>\d+)\s+\$?(?P<unit_price>[0-9,]+(?:\.\d{1,2})?)\s+\$?(?P<line_total>[0-9,]+(?:\.\d{1,2})?)"
        )

        for match in pattern.finditer(text):
            item = match.group("item").strip()
            qty = int(match.group("qty"))
            unit_price = float(match.group("unit_price").replace(",", ""))
            total = float(match.group("line_total").replace(",", ""))

            invoice_total += total
            line_items.append({
                "item": item,
                "qty": qty,
                "unit_price": unit_price,
                "total": total
            })

        # If we didn’t find any line items, try to find a “Total Due” number:
        if not line_items:
            total_match = re.search(r"Total\s+Due[:\s]+\$?([0-9,]+(?:\.\d{1,2})?)", text, re.IGNORECASE)
            if total_match:
                invoice_total = float(total_match.group(1).replace(",", ""))

        return line_items, invoice_total

    def _find_policies(self, text: str) -> List[str]:
        """
        Return a list of all policy keywords from self.policy_keywords
        found in the PDF text (case-insensitive).
        """
        found = []
        for kw in self.policy_keywords:
            if re.search(rf"\b{kw}\b", text, re.IGNORECASE):
                found.append(kw)
        return found
