# agents/json_agent.py

import os
import json
from typing import Any, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field

load_dotenv()

class FlowBitSchema(BaseModel):
    """
    Define your FlowBit schema here. 
    Adjust the field names and types to your actual JSON structure.
    Example fields below:
    """
    order_id: str = Field(..., description="Unique order identifier")
    product: str = Field(..., description="Name of the product")
    quantity: int = Field(..., description="Number of units being ordered")
    price: float = Field(..., description="Price per unit")
    # You can add more fields, e.g.
    # customer_email: str
    # delivery_date: datetime
    # etc.

class JSONAgent:
    """
    JSONAgent:
      - Parses raw JSON bytes → a Python dict
      - Validates against FlowBitSchema
      - Flags anomalies (missing or wrong‑type fields)
      - Suggests action: "alert" → risk_alert if invalid, otherwise "store" → database
    """

    def __init__(self):
        # This agent does not need an LLM by default
        pass

    def process(self, raw_bytes: bytes, metadata: Dict[str, Any]) -> dict:
        """
        1) Decode raw_bytes → Python dict
           If decode fails → valid=False, errors=[…], suggestion={"action":"alert","target":"risk_alert"}
        2) Validate with Pydantic (FlowBitSchema)
           If ValidationError → valid=False, errors=[…]
           else → valid=True, errors=[]
        3) Build data dict + choose action_suggestion
        4) Return:
           { "source":"json_agent", "data":{…}, "action_suggestion":{…} }
        """
        # 1) Parse JSON
        try:
            payload_str = raw_bytes.decode("utf-8")
            payload_dict = json.loads(payload_str)
        except Exception as e:
            data = {
                "parsed_json": None,
                "valid": False,
                "errors": [f"JSONParseError: {str(e)}"]
            }
            action_suggestion = {"action": "alert", "target": "risk_alert"}
            return {
                "source": "json_agent",
                "data": data,
                "action_suggestion": action_suggestion
            }

        # 2) Validate schema
        try:
            validated = FlowBitSchema(**payload_dict)
            valid = True
            errors = []
        except ValidationError as ve:
            valid = False
            # Pydantic’s .errors() gives a list of dicts, but we can stringify for simplicity
            errors = [err["msg"] for err in ve.errors()]

        data = {
            "parsed_json": payload_dict,
            "valid":        valid,
            "errors":       errors
        }

        # 3) Decide action
        if not valid:
            action_suggestion = {"action": "alert", "target": "risk_alert"}
        else:
            action_suggestion = {"action": "store", "target": "database"}

        return {
            "source": "json_agent",
            "data": data,
            "action_suggestion": action_suggestion
        }
