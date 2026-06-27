from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from alien_inference import AlienInference
import os

app = FastAPI(title="Alien-LLM Intelligence Interface")

# Initialize model
model = None
if os.path.exists("model_checkpoint.bin") and os.path.exists("vocab.txt"):
    model = AlienInference("vocab.txt", "model_checkpoint.bin")
else:
    print("[WARNING] Model or Vocab not found. Interface will be in mock mode.")

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("interface.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat(req: ChatRequest):
    if model:
        try:
            response = model.generate(req.message)
            return {"response": response}
        except Exception as e:
            return {"response": f"Error: {str(e)}"}
    return {"response": "The Alien Intelligence has not been trained yet. Please run the training pipeline first!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
