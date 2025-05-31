# action_router.py

import os
import asyncio
from datetime import datetime

import httpx  # lightweight async HTTP client; pip install httpx

class ActionRouter:
    """
    ActionRouter:
      - Takes in an `action_suggestion` dict from an agent: 
          { "action": str, "target": str }
      - Maps `target` → a local (simulated) endpoint:
          - "crm"        → POST /crm
          - "risk_alert" → POST /risk_alert
          - "database"   → just log (or write to memory directly)
      - Sends an HTTP request with minimal retry logic
      - Returns a standardized outcome dict:
          {
            "status": "success" | "error",
            "target": <str>,
            "http_status": <int>,
            "response_body": <dict>,
            "error": <optional str>
          }
    """

    def __init__(self, base_url: str = None):
        """
        base_url: The host where /crm and /risk_alert are served, 
                  e.g., "http://localhost:8000". 
                  Default: read from ENV or fallback to localhost:8000.
        """
        env_url = os.getenv("BASE_URL", None)
        self.base_url = base_url or env_url or "http://localhost:8000"

        # An AsyncClient lets us do connection pooling and reuse
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=10.0)

    async def decide_and_execute(self, suggestion: dict) -> dict:
        """
        suggestion: { "action": <str>, "target": <str> }
        - Maps `target` → a path ("/crm", "/risk_alert", or just log for "database")
        - Builds a JSON payload: { "action": action, "timestamp": ... }
        - Calls `_post_with_retries(...)` for HTTP‑based targets
        - Returns a standardized outcome dict
        """
        action = suggestion.get("action")
        target = suggestion.get("target")

        # Build a simple payload
        payload = {
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }

        # 1) CRM endpoint
        if target == "crm":
            path = "/crm"
        # 2) Risk Alert endpoint
        elif target == "risk_alert":
            path = "/risk_alert"
        # 3) Database / Archive
        elif target == "database":
            # For “archive,” we don’t need an HTTP call—
            # we can simply return success and let the orchestrator or agent write to memory/db itself.
            return {
                "status": "success",
                "target": "database",
                "http_status": None,
                "response_body": {"message": "Archived to database."},
                "error": None
            }
        else:
            return {
                "status": "error",
                "target": target,
                "http_status": None,
                "response_body": None,
                "error": f"Unknown target: {target}"
            }

        # 4) Make the HTTP call with retry logic
        result = await self._post_with_retries(path, payload)
        return result

    async def _post_with_retries(self, path: str, payload: dict, max_retries: int = 2) -> dict:
        """
        Attempt an asynchronous POST to `self.base_url + path` up to (max_retries + 1) times.
        Returns a dict:
          {
            "status": "success"|"error",
            "target": <the path without leading "/">,
            "http_status": <int or None>,
            "response_body": <dict or None>,
            "error": <str or None>
          }
        """
        last_error = None
        http_status = None
        response_body = None

        for attempt in range(max_retries + 1):
            try:
                resp = await self.client.post(path, json=payload)
                http_status = resp.status_code
                resp.raise_for_status()  # raise an exception if 4xx/5xx
                response_body = resp.json()
                return {
                    "status": "success",
                    "target": path.lstrip("/"),
                    "http_status": http_status,
                    "response_body": response_body,
                    "error": None
                }
            except Exception as e:
                last_error = str(e)
                # Simple backoff before retrying
                await asyncio.sleep(1)

        # If all retries fail, return an error dict
        return {
            "status": "error",
            "target": path.lstrip("/"),
            "http_status": http_status,
            "response_body": response_body,
            "error": last_error
        }

    async def shutdown(self):
        """
        Call this when your app stops to close the HTTP client cleanly.
        """
        await self.client.aclose()
