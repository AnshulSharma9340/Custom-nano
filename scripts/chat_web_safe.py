#!/usr/bin/env python3
"""
Safe Medical Web Chat Server
Serves both UI and API from a single FastAPI instance.
Includes an integrated System Prompt Safety Layer.
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0 
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description='NanoChat Web Server - Safe Edition')
parser.add_argument('-n', '--num-gpus', type=int, default=1)
parser.add_argument('-i', '--source', type=str, default="sft")
parser.add_argument('-t', '--temperature', type=float, default=0.8)
parser.add_argument('-k', '--top-k', type=int, default=50)
parser.add_argument('-m', '--max-tokens', type=int, default=512)
parser.add_argument('-g', '--model-tag', type=str, default=None)
parser.add_argument('-s', '--step', type=int, default=None)
parser.add_argument('-p', '--port', type=int, default=8000)
parser.add_argument('--device-type', type=str, default='')
parser.add_argument('--host', type=str, default='0.0.0.0')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

@dataclass
class Worker:
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object

class WorkerPool:
    def __init__(self, num_gpus=None):
        self.num_gpus = num_gpus if num_gpus else (torch.cuda.device_count() if device_type == "cuda" else 1)
        self.workers, self.available_workers = [], asyncio.Queue()

    async def initialize(self, source, model_tag=None, step=None):
        for gpu_id in range(self.num_gpus):
            dev = torch.device(f"cuda:{gpu_id}" if device_type == "cuda" else device_type)
            model, tokenizer, _ = load_model(source, dev, phase="eval", model_tag=model_tag, step=step)
            worker = Worker(gpu_id, dev, Engine(model, tokenizer), tokenizer)
            self.workers.append(worker)
            await self.available_workers.put(worker)

    async def acquire_worker(self): return await self.available_workers.get()
    async def release_worker(self, worker): await self.available_workers.put(worker)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None

def validate_chat_request(request: ChatRequest):
    if len(request.messages) == 0: raise HTTPException(status_code=400, detail="At least one message is required")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    with open(os.path.join("nanochat", "ui.html"), "r", encoding="utf-8") as f: content = f.read()
    return HTMLResponse(content=content.replace("const API_URL = `http://${window.location.hostname}:8000`;", "const API_URL = '';"))

@app.get("/logo.svg")
async def logo(): return FileResponse(os.path.join("nanochat", "logo.svg"), media_type="image/svg+xml")

async def generate_stream(worker, tokens, temperature=None, max_new_tokens=None, top_k=None):
    accumulated_tokens, last_clean_text = [], ""
    for token_column, _ in worker.engine.generate(tokens, num_samples=1, max_tokens=max_new_tokens or args.max_tokens, temperature=temperature or args.temperature, top_k=top_k or args.top_k, seed=random.randint(0, 2**31 - 1)):
        token = token_column[0]
        if token in [worker.tokenizer.encode_special("<|assistant_end|>"), worker.tokenizer.get_bos_token_id()]: break
        
        accumulated_tokens.append(token)
        current_text = worker.tokenizer.decode(accumulated_tokens)
        if not current_text.endswith(''):
            if new_text := current_text[len(last_clean_text):]:
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text
    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    validate_chat_request(request)

    # --- INVISIBLE SYSTEM PROMPT LAYER ---
    SYSTEM_PROMPT = """You are a safe and accurate medical information assistant.
You follow these rules strictly:
- Always recommend consulting a licensed doctor for personal medical decisions
- Never provide prescriptions, diagnoses, or treatment plans
- If you are unsure about something, say so clearly
- In emergencies, always direct to emergency services first
- Never provide information that could be used to cause harm
- When you don't know something, say: "I don't have reliable information on this. Please consult a medical professional."
"""
    
    if len(request.messages) > 0 and request.messages[0].role == "user":
        # Check if we've already injected the prompt to avoid stacking it on every message
        if "You are a safe and accurate medical" not in request.messages[0].content:
            request.messages[0].content = f"{SYSTEM_PROMPT}\n\nPatient Query: {request.messages[0].content}"
    # -------------------------------------

    worker = await app.state.worker_pool.acquire_worker()
    
    try:
        tokens = [worker.tokenizer.get_bos_token_id()]
        for m in request.messages:
            tokens.extend([worker.tokenizer.encode_special(f"<|{m.role}_start|>")] + worker.tokenizer.encode(m.content) + [worker.tokenizer.encode_special(f"<|{m.role}_end|>")])
        tokens.append(worker.tokenizer.encode_special("<|assistant_start|>"))

        async def stream_and_release():
            try:
                async for chunk in generate_stream(worker, tokens, request.temperature, request.max_tokens, request.top_k): yield chunk
            finally: await app.state.worker_pool.release_worker(worker)
        return StreamingResponse(stream_and_release(), media_type="text/event-stream")
    except Exception as e:
        await app.state.worker_pool.release_worker(worker)
        raise e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)