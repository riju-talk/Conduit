import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        return f"Error: {str(e)}"

def is_json(data):
    """Check if a string is valid JSON."""
    try:
        import json
        json.loads(data)
        return True
    except:
        return False

def classify_intent(text):
    """Simple intent classification (extendable with AI)."""
    if "invoice" in text.lower():
        return "Invoice"
    elif "rfq" in text.lower():
        return "RFQ"
    else:
        return "Complaint"