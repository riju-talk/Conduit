from fastapi import FastAPI, UploadFile, File, HTTPException
from agents.classifier import ClassifierAgent
from agents.email_agent      import EmailAgent
from agents.json_agent       import JSONAgent
from agents.pdf_agent        import PDFAgent
from memory.memory            import MemoryStore
from mcp.router              import ActionRouter
import logging
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Conduit starting up: initializing components")
    app.state.memory       = MemoryStore()
    app.state.classifier   = ClassifierAgent()
    app.state.email_agent  = EmailAgent()
    app.state.json_agent   = JSONAgent()
    app.state.pdf_agent    = PDFAgent()
    app.state.router       = ActionRouter()
    yield  # everything after this is shutdown logic

        # --- SHUTDOWN LOGIC ---
    logging.info("Conduit shutting down: cleaning up resources")
        # Close memory connections
    try:
        app.state.memory.close()      # e.g., redis.close() or sqlite connection close
    except Exception as e:
        logging.warning(f"Error closing memory: {e}")


app = FastAPI(title="Conduit")
app = FastAPI(lifespan=lifespan)

@app.get("/", tags=["Health"])
async def running_message():
    return {"message": "Conduit service is running; ready to process inputs."}

# === Core Ingest Route ===

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()

    # 1) Classify
    meta = app.state.classifier.process(raw)
    app.state.memory.write("classifier", "metadata", meta)

    # 2) Dispatch
    fmt = meta["format"]
    if fmt == "Email":
        result = app.state.email_agent.process(raw, meta)
    elif fmt == "JSON":
        result = app.state.json_agent.process(raw, meta)
    elif fmt == "PDF":
        result = app.state.pdf_agent.process(raw, meta)
    else:
        raise HTTPException(status_code=400, detail="Unknown format")

    app.state.memory.write(result["source"], "extraction", result["data"])

    # 3) Route action
    action_outcome = await app.state.router.decide_and_execute(result["action_suggestion"])
    app.state.memory.write("router", "action", action_outcome)

    return {
        "metadata": meta,
        "extraction": result["data"],
        "action": action_outcome
    }

@app.post("/crm")
async def crm_escalation(payload: dict):
    # Simulate creating a CRM ticket
    return {"status": "success", "detail": {"message": "CRM ticket created."}}

@app.post("/risk_alert")
async def risk_alert(payload: dict):
    # Simulate logging a risk alert
    return {"status": "success", "detail": {"message": "Risk alert logged."}}


@app.get("/memory", tags=["Memory"])
async def read_memory():
    """
    Dump the shared memory events for audit/debug.
    """
    # return app.state.memory.read_all()
    return {"events": []}  # stub

# === Run Server ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
