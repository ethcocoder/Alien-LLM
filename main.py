import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from interface import AlienLLMInterface

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = None
try:
    LIB_PATH = os.path.join(os.path.dirname(__file__), "build/libalien_llm_lib.so")
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "model_checkpoint.bin")
    if os.path.exists(LIB_PATH):
        model = AlienLLMInterface(LIB_PATH, CHECKPOINT_PATH)
    else:
        print(f"Warning: Shared library not found at {LIB_PATH}")
except Exception as e:
    print(f"Error initializing model: {e}")

# Serve static files
# For Vercel, we mount the 'frontend' directory as 'static'
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("frontend/index.html", "r") as f:
        return f.read()

@app.get("/documentation", response_class=HTMLResponse)
async def read_docs():
    with open("frontend/documentation.html", "r") as f:
        return f.read()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("message", "")
    
    async def event_generator():
        if model:
            for token in model.generate(prompt, max_tokens=50):
                yield f"data: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0.05) # Simulate streaming delay
        else:
            yield f"data: {json.dumps({'token': 'Error: Model not initialized. Shared library may be missing.'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
