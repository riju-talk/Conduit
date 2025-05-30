import os
import email
from email import policy
from email.parser import BytesParser



"""
structure: request and response

request:




response:
extracted_fields = {
    "sender:"
    "urgency:
    "issue:"
    "tone:"
    "
}


"""
def parse_email_file(file_path):
    """
    Parse an email file (.eml, .txt, etc.) and return the email body as text.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.eml', '.msg', '.txt']:
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        # Try to get the plain text part
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    return part.get_content()
        else:
            return msg.get_content()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def email_parser_agent(text, intent):
    """
    Parse the email text for fields based on the intent schema.
    """
    schema = get_schema_for_intent(intent)
    extracted_fields = {}

    # Simple extraction logic (extend with LLM if needed)
    for field in schema:
        if field in text.lower():
            start = text.lower().find(field) + len(field) + 1
            end = text.find('\n', start) if '\n' in text[start:] else len(text)
            extracted_fields[field] = text[start:end].strip()
        else:
            extracted_fields[field] = "Not found"

    return extracted_fields

def get_schema_for_intent(intent):
    """Return schema for the given intent."""
    schemas = {
        'Invoice': ['invoice_number', 'date', 'total'],
        'RFQ': ['requested_items', 'deadline'],
        'Complaint': ['issue', 'customer_name']
    }
    return schemas.get(intent, [])