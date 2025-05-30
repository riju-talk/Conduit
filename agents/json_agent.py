import json

def json_agent(json_data, intent):
    """Extract fields from JSON based on intent."""
    data = json.loads(json_data)
    schema = get_schema_for_intent(intent)
    extracted_fields = {}
    for field in schema:
        extracted_fields[field] = data.get(field)
    return extracted_fields

def get_schema_for_intent(intent):
    """Return schema for the given intent."""
    schemas = {
        'Invoice': ['invoice_number', 'date', 'total'],
        'RFQ': ['requested_items', 'deadline'],
        'Complaint': ['issue', 'customer_name']
    }
    return schemas.get(intent, [])