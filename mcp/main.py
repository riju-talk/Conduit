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

@app.get("/")
async def health_check():
    return {"message": "Conduit service is running; ready to process files."}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    1) Read raw bytes
    2) Classify format + intent
    3) Dispatch to appropriate agent
    4) Write metadata + extraction to memory
    5) Route the suggested action
    6) Write action outcome to memory
    7) Return combined result
    """
    raw_bytes = await file.read()

    # Step 1: Classify
    metadata = app.state.classifier.process(raw_bytes, file.filename)
    app.state.memory.write("classifier", "metadata", metadata)

    # Step 2: Dispatch
    fmt = metadata.get("format", "")
    if fmt == "Email":
        result = app.state.email_agent.process(raw_bytes, metadata)
    elif fmt == "JSON":
        result = app.state.json_agent.process(raw_bytes, metadata)
    elif fmt == "PDF":
        result = app.state.pdf_agent.process(raw_bytes, metadata)
    else:
        raise HTTPException(status_code=400, detail="Unknown format")

    # Step 3: Persist extraction
    app.state.memory.write(result["source"], "extraction", result["data"])

    # Step 4: Route action
    action_outcome = await app.state.router.decide_and_execute(result["action_suggestion"])
    app.state.memory.write("router", "action", action_outcome)

    # Step 5: Return to client
    return {
        "metadata": metadata,
        "extraction": result["data"],
        "action": action_outcome
    }

# Simulated endpoints for /crm and /risk_alert:
@app.post("/crm")
async def crm_escalation(payload: dict):
    # In prod, you'd call a CRM API. Here, we just echo a “ticket created” response.
    return {"status": "escalate", "detail": {"message": "CRM ticket created."}}

@app.post("/risk_alert")
async def risk_alert(payload: dict):
    # Simulate flagging a compliance/fraud risk
    return {"status": "escalate", "detail": {"message": "Risk alert logged."}}

@app.get("/audit")
async def audit():
    """
    Retrieve all events from the MemoryStore where key is 'action'.
    """
    actions = app.state.memory.read_by_key("action")
    return {"actions": actions}

@app.get("/audit/store")
async def audit_store():
    """
    Retrieve events where the action type is 'store'.
    """
    actions = app.state.memory.read_by_key("action")
    store_actions = [event for event in actions if event["value"].get("action") == "store"]
    return {"store_actions": store_actions}

@app.get("/audit/alert")
async def audit_alert():
    """
    Retrieve events where the action type is 'alert'.
    """
    actions = app.state.memory.read_by_key("action")
    alert_actions = [event for event in actions if event["value"].get("action") == "alert"]
    return {"alert_actions": alert_actions}

@app.get("/audit/escalate")
async def audit_escalate():
    """
    Retrieve events where the action type is 'escalate'.
    """
    actions = app.state.memory.read_by_key("action")
    escalate_actions = [event for event in actions if event["value"].get("action") == "escalate"]
    return {"escalate_actions": escalate_actions}

@app.get("/audit/log")
async def audit_log():
    """
    Retrieve events where the action type is 'log'.
    """
    actions = app.state.memory.read_by_key("action")
    log_actions = [event for event in actions if event["value"].get("action") == "log"]
    return {"log_actions": log_actions}


# === Run Server ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
