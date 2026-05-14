import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import torch

from backend.optimizer import run_optimization_loop

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

import subprocess

@app.get("/extract_formula")
async def extract_formula():
    # Run the PySR extractor in the background so we don't block the API
    subprocess.Popen(["python", "run_pysr_extractor.py"])
    return {"status": "success", "message": "PySR extraction triggered in background."}

@app.websocket("/ws/synthesize")
async def websocket_synthesize(websocket: WebSocket):
    await websocket.accept()
    try:
        config = await websocket.receive_text()
        params = json.loads(config)
        await run_optimization_loop(websocket, params)
    except WebSocketDisconnect:
        print("Client disconnected.")
    except torch.cuda.OutOfMemoryError as e:
        import traceback
        import sys
        traceback.clear_frames(e.__traceback__)
        del e
        print("CUDA ERROR: Out of memory during synthesis! Falling back to CPU...")
        await websocket.send_json({"type": "log", "message": "CUDA OOM! GPU memory exhausted. Falling back to CPU mode. This will be slower but will complete..."})
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        try:
            await run_optimization_loop(websocket, params, force_cpu=True)
        except Exception as e2:
            print(f"Error during CPU fallback: {e2}")
            try:
                await websocket.send_json({"type": "error", "message": f"CPU Fallback failed: {e2}"})
            except:
                pass
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        print("connection closed")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
