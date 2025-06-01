import os
import json
import logging
from typing import Any, Dict, List, Union
from datetime import datetime
from jsonschema import validate, ValidationError

class JSONAgent:
    """
    Simplified JSONAgent that:
      - Parses raw JSON bytes or string
      - Validates against a predefined schema based on `metadata["intent"]`
      - Detects a couple of basic anomalies (e.g., high amounts, missing fields)
      - Decides on an action_suggestion
      - Logs a summary to shared_memory if provided
    """

    def __init__(self, shared_memory=None):
        self.shared_memory = shared_memory
        self.logger = logging.getLogger(__name__)

        # Only the schemas your project currently needs—others can be added similarly.
        self.schemas = {
            "rfq": {
                "type": "object",
                "required": ["rfq_id", "company", "items", "deadline"],
                "properties": {
                    "rfq_id": {"type": "string"},
                    "company": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["item_name", "quantity"],
                            "properties": {
                                "item_name": {"type": "string"},
                                "quantity": {"type": "number", "minimum": 1}
                            }
                        }
                    },
                    "deadline": {"type": "string"},
                    "budget_range": {"type": "number", "minimum": 0}
                }
            },
            "complaint": {
                "type": "object",
                "required": ["complaint_id", "customer_id", "issue_type", "description"],
                "properties": {
                    "complaint_id": {"type": "string"},
                    "customer_id": {"type": "string"},
                    "issue_type": {"type": "string", "enum": ["product", "service", "billing", "delivery", "other"]},
                    "description": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                }
            },
            "fraud_risk": {
                "type": "object",
                "required": ["transaction_id", "amount", "user_id"],
                "properties": {
                    "transaction_id": {"type": "string"},
                    "amount": {"type": "number", "minimum": 0},
                    "user_id": {"type": "string"},
                    "risk_score": {"type": "number", "minimum": 0, "maximum": 100}
                }
            },
            "webhook": {
                "type": "object",
                "required": ["event_type", "timestamp", "data"],
                "properties": {
                    "event_type": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "data": {"type": "object"}
                }
            }
        }

    def process(self, json_data: Union[str, bytes, dict], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        1) Parse raw JSON (bytes or string) into a Python dict
        2) Validate against the schema for metadata["intent"]
        3) Run simple anomaly checks:
           - For "rfq": very large budget_range or item quantity
           - For "fraud_risk": high amount or high risk_score
        4) Decide an action_suggestion
        5) Return a standardized dict and log to shared_memory
        """
        try:
            # 1) Parse JSON
            if isinstance(json_data, (bytes, bytearray)):
                text = json_data.decode("utf-8")
                parsed = json.loads(text)
            elif isinstance(json_data, str):
                parsed = json.loads(json_data)
            else:
                parsed = json_data

            intent = metadata.get("intent", "webhook").lower()
            source_id = metadata.get("source_id", f"json_{datetime.now().timestamp()}")

            # 2) Validate against schema (if available)
            schema = self.schemas.get(intent)
            validation = {"is_valid": False, "errors": []}
            if schema:
                try:
                    validate(instance=parsed, schema=schema)
                    validation["is_valid"] = True
                except ValidationError as ve:
                    validation["errors"].append(ve.message)
            else:
                # No schema defined for this intent
                validation["errors"].append(f"No schema for intent '{intent}'")

            # 3) Simple anomaly detection
            anomalies: List[Dict[str, Any]] = []
            if intent == "rfq" and isinstance(parsed, dict):
                budget = parsed.get("budget_range", 0)
                if isinstance(budget, (int, float)) and budget > 100000:
                    anomalies.append({"type": "high_budget", "value": budget, "severity": "medium"})
                items = parsed.get("items", [])
                if isinstance(items, list):
                    for idx, item in enumerate(items):
                        qty = item.get("quantity", 0)
                        if isinstance(qty, (int, float)) and qty > 10000:
                            anomalies.append({"type": "unrealistic_quantity", "item_index": idx, "value": qty, "severity": "medium"})
            elif intent == "fraud_risk" and isinstance(parsed, dict):
                amount = parsed.get("amount", 0)
                if isinstance(amount, (int, float)) and amount > 50000:
                    anomalies.append({"type": "large_transaction", "value": amount, "severity": "high"})
                risk = parsed.get("risk_score", 0)
                if isinstance(risk, (int, float)) and risk > 70:
                    anomalies.append({"type": "high_risk_score", "value": risk, "severity": "critical"})

            # 4) Extract key fields (simple subset)
            extracted_fields = {}
            try:
                if intent == "rfq":
                    extracted_fields = {
                        "rfq_id": parsed.get("rfq_id"),
                        "company": parsed.get("company"),
                        "item_count": len(parsed.get("items", []))
                    }
                elif intent == "complaint":
                    extracted_fields = {
                        "complaint_id": parsed.get("complaint_id"),
                        "issue_type": parsed.get("issue_type"),
                        "severity": parsed.get("severity", "medium")
                    }
                elif intent == "fraud_risk":
                    extracted_fields = {
                        "transaction_id": parsed.get("transaction_id"),
                        "amount": parsed.get("amount"),
                        "risk_score": parsed.get("risk_score", 0)
                    }
                else:
                    extracted_fields = {"summary": "No key fields defined for this intent."}
            except Exception as e:
                self.logger.error(f"Error extracting key fields: {e}")

            # 5) Decide action
            if anomalies:
                top_severity = max(a["severity"] for a in anomalies if "severity" in a)
                if top_severity in ("critical", "high"):
                    action = {"action": "flag_for_review", "target": "risk_team", "reason": "Critical/High anomaly"}
                else:
                    action = {"action": "log_anomalies", "target": "monitoring", "reason": "Minor anomalies detected"}
            elif not validation["is_valid"]:
                action = {"action": "reject", "target": "error_queue", "reason": "Schema validation failed"}
            else:
                action = {"action": "process_normally", "target": f"{intent}_handler", "reason": "Valid data"}

            # Build response
            response = {
                "source": "json_agent",
                "source_id": source_id,
                "timestamp": datetime.now().isoformat(),
                "intent": intent,
                "validation": validation,
                "anomalies": anomalies,
                "extracted_fields": extracted_fields,
                "data": parsed,
                "action_suggestion": action,
                "status": "success"
            }

            # 6) Log a summary into shared_memory if available
            if self.shared_memory:
                try:
                    summary = {
                        "agent": "json_agent",
                        "timestamp": response["timestamp"],
                        "intent": intent,
                        "valid": validation["is_valid"],
                        "anomaly_count": len(anomalies),
                        "action": action["action"]
                    }
                    self.shared_memory.store("json_agent_log", summary)
                except Exception as e:
                    self.logger.error(f"Shared memory log failed: {e}")

            return response

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            return {
                "source": "json_agent",
                "source_id": metadata.get("source_id", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": f"Invalid JSON: {e}",
                "validation": {"is_valid": False, "errors": [str(e)]},
                "anomalies": [],
                "extracted_fields": {},
                "data": {},
                "action_suggestion": {"action": "reject", "target": "error_queue", "reason": "Invalid JSON"}
            }
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return {
                "source": "json_agent",
                "source_id": metadata.get("source_id", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "validation": {"is_valid": False, "errors": [str(e)]},
                "anomalies": [],
                "extracted_fields": {},
                "data": {},
                "action_suggestion": {"action": "reject", "target": "error_queue", "reason": "Processing exception"}
            }
