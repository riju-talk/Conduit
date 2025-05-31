import os
import json
from datetime import datetime
import redis

class MemoryStore:
    """
    A simple Redis‑based “blackboard” where every agent (or router) can append an event.
    Each event is a JSON blob with:
      - timestamp
      - source    (e.g., "classifier", "email_agent", "router")
      - key       (e.g., "metadata", "extraction", "action")
      - value     (a dict of agent‑specific data)
    """

    def __init__(self, host: str = None, port: int = None, db: int = 0):
        # Read from environment (so it works in Docker Compose, etc.)
        redis_host = host or os.getenv("REDIS_HOST", "localhost")
        redis_port = port or int(os.getenv("REDIS_PORT", 6379))
        self.client = redis.Redis(host=redis_host, port=redis_port, db=db, decode_responses=True)
        self.list_key = "memory:events"

    def write(self, source: str, key: str, value: dict):
        """
        Append a new event to the memory.
        - source: which agent or component ("classifier", "email_agent", "router", etc.)
        - key: short tag ("metadata", "extraction", "action")
        - value: a JSON‑serializable dict with whatever data you need to store
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "source":    source,
            "key":       key,
            "value":     value
        }
        # Push to the right (newest at the end)
        self.client.rpush(self.list_key, json.dumps(event))

    def read_all(self) -> list:
        """
        Return all events in chronological order (oldest → newest).
        """
        raw = self.client.lrange(self.list_key, 0, -1)
        return [json.loads(record) for record in raw]

    def read_by_source(self, source: str) -> list:
        """
        Return only those events where event["source"] == source.
        """
        return [e for e in self.read_all() if e["source"] == source]

    def read_by_key(self, key: str) -> list:
        """
        Return only those events where event["key"] == key.
        """
        return [e for e in self.read_all() if e["key"] == key]

    def close(self):
        """
        Gracefully close the Redis connection (optional).
        """
        try:
            self.client.close()
        except Exception:
            pass
