import redis
import json

client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

def save_to_memory(input_id, metadata, extracted_fields=None):
    """Saving metadata and extracted fields to Redis."""
    data = {
        "metadata": metadata,
        "extracted_fields": extracted_fields or {}
    }
    client.set(input_id, json.dumps(data))

def get_from_memory(input_id):
    """Retrieving data from Redis using input_id."""
    data = client.get(input_id)
    return json.loads(data) if data else {}