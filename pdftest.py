from PyPDF2 import PdfReader

reader = PdfReader("data/invoice.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() or ""
    if len(text.split()) >= 10:
        break
words = text.split()
print(" ".join(words[:10]))