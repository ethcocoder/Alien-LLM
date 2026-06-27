import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from alien_inference import AlienInference

app = FastAPI(title="Alien Intelligence (AI2)")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model using pure Python inference (no C++ needed)
model = None
print("--- Alien-LLM Startup (Pure Python Mode) ---")
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VOCAB_PATH = os.path.join(BASE_DIR, "vocab.txt")
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "model_checkpoint.bin")

    if os.path.exists(VOCAB_PATH) and os.path.exists(CHECKPOINT_PATH):
        print(f"[INFO] Loading vocab from: {VOCAB_PATH}")
        print(f"[INFO] Loading checkpoint from: {CHECKPOINT_PATH}")
        model = AlienInference(VOCAB_PATH, CHECKPOINT_PATH)
        print("[SUCCESS] Model initialized successfully (Pure Python).")
    else:
        missing = []
        if not os.path.exists(VOCAB_PATH): missing.append("vocab.txt")
        if not os.path.exists(CHECKPOINT_PATH): missing.append("model_checkpoint.bin")
        print(f"[ERROR] Missing files: {', '.join(missing)}")
except Exception as e:
    print(f"[CRITICAL] Error initializing model: {e}")
    import traceback
    traceback.print_exc()
print("---------------------------------------------")

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(FRONTEND_DIR, "index.html"), "r") as f:
        return f.read()

@app.get("/documentation", response_class=HTMLResponse)
async def read_docs():
    with open(os.path.join(FRONTEND_DIR, "documentation.html"), "r") as f:
        return f.read()

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        prompt = data.get("message", "")
        print(f"[CHAT] Received: {prompt[:80]}")

        async def event_generator():
            if model:
                try:
                    print(f"[CHAT] Generating response for: {prompt[:30]}...")
                    token_count = 0
                    for token in model.generate_stream(prompt, max_len=50):
                        yield f"data: {json.dumps({'token': token})}\n\n"
                        token_count += 1
                        await asyncio.sleep(0.02) # Snappy streaming 
                    print(f"[CHAT] Generation complete. Tokens: {token_count}")
                except Exception as e:
                    print(f"[ERROR] Generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    yield f"data: {json.dumps({'token': f'Error: {str(e)}'})}\n\n"
            else:
                yield f"data: {json.dumps({'token': 'Error: Model not loaded. Check server logs.'})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        print(f"[ERROR] Chat endpoint error: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mode": "pure-python"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
