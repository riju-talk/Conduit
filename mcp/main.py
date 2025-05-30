from fastapi import FastAPI, UploadFile, File, Form
from agents.classifier import classifier_agent
import os

app = FastAPI(title="Conduit")



@app.lifespan("startup")



@app.lifespan("shutdown")


@app.get("/")
async def running_message():
    #returning a message to indicate the service is running
    return {"message": "Conduit service is running; ready to process files and text inputs."}

@app.post("/upload")


@app.post("/crm")


@app.post("/risk_alert")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)