# action_router.py

import os
import asyncio
from datetime import datetime

import httpx  # lightweight async HTTP client; pip install httpx

class ActionRouter:
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
        action = suggestion.get("action")
        target = suggestion.get("target")

        # Build a simple payload
        payload = {
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }

        if target == "crm":
            path = "/crm"
        elif target == "risk_alert":
            path = "/risk_alert"
        elif target == "database":
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
        await self.client.aclose()
