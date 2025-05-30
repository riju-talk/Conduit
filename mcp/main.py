from fastapi import FastAPI, UploadFile, File, Form
from agents.classifier import classifier_agent
import os

app = FastAPI(title="Conduit")

@app.post("/process")
async def process_input(file: UploadFile = File(None), text: str = Form(None)):
    """Process input file or text and return extracted data."""
    if file:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        input_id, extracted_fields = classifier_agent(None, file_path=file_path)
        os.remove(file_path)  # Clean up temporary file
    elif text:
        input_id, extracted_fields = classifier_agent(text)
    else:
        return {"error": "No input provided"}

    return {"input_id": input_id, "extracted_fields": extracted_fields}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)