#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.
Includes Data Parallelism for the heavy model AND an integrated DistilBERT Safety Guardrail.

Launch examples:
  # 2 GPUs with the DistilBERT safety model
  python -m scripts.chat_web_safe -n 2 --safety-repo anshul32467/Medical-DistilBert --host 0.0.0.0 --port 8090
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

# Use AutoTokenizer so any tokenizer variant saved in the checkpoint works
from transformers import AutoTokenizer, DistilBertForSequenceClassification

# ---------------------------------------------------------------------------
# Abuse-prevention limits
# ---------------------------------------------------------------------------
MAX_MESSAGES_PER_REQUEST       = 500
MAX_MESSAGE_LENGTH             = 8000
MAX_TOTAL_CONVERSATION_LENGTH  = 32000
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.0, 2.0
MIN_TOP_K,       MAX_TOP_K       = 0,   200   # 0 = disable top-k (full vocab)
MIN_MAX_TOKENS,  MAX_MAX_TOKENS  = 1,   4096

# ---------------------------------------------------------------------------
# Label sets your DistilBERT safety model may use.
# We normalise every predicted label to UPPER-CASE before matching,
# so "self_harm", "SELF_HARM", "Self Harm" all work.
#
# *** After the server starts, check the log line:
#       "Safety model labels: {…}"
#     and update these sets if your model uses different strings. ***
# ---------------------------------------------------------------------------
SELF_HARM_LABELS     = {"SELF_HARM", "SELF-HARM", "SELFHARM", "SUICIDE", "SELF_INJURY"}
DANGEROUS_LABELS     = {"DANGEROUS", "DANGER", "HARMFUL", "VIOLENCE", "TOXIC"}
MISINFORMATION_LABELS = {"MISINFORMATION", "MISINFO", "FAKE", "FALSE", "MISLEADING"}
# Any label NOT in the above sets (and not SAFE/NORMAL/BENIGN) will be
# logged as a warning and the request will be allowed through — change
# this behaviour below if you prefer a stricter default.
SAFE_LABELS          = {"SAFE", "NORMAL", "BENIGN", "LEGITIMATE", "GENERAL",
                        "LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3", "LABEL_4"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="NanoChat Web Server with DistilBERT Safety Guardrail")
parser.add_argument("-n", "--num-gpus",    type=int,   default=1,     help="Number of GPUs (default: 1)")
parser.add_argument("-i", "--source",      type=str,   default="sft", help="Model source: sft|rl")
parser.add_argument("-t", "--temperature", type=float, default=0.8,   help="Default sampling temperature")
parser.add_argument("-k", "--top-k",       type=int,   default=50,    help="Default top-k value")
parser.add_argument("-m", "--max-tokens",  type=int,   default=512,   help="Default max new tokens")
parser.add_argument("-g", "--model-tag",   type=str,   default=None,  help="Model tag to load")
parser.add_argument("-s", "--step",        type=int,   default=None,  help="Checkpoint step to load")
parser.add_argument("-p", "--port",        type=int,   default=8000,  help="Server port")
parser.add_argument("--device-type",       type=str,   default="",
                    choices=["cuda", "cpu", "mps"],
                    help="Device type. Empty = autodetect")
parser.add_argument("--host",              type=str,   default="0.0.0.0", help="Bind host")
parser.add_argument("--safety-repo",       type=str,   default="anshul32467/Medical-DistilBert",
                    help="HuggingFace repo for the DistilBERT safety guardrail")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device init
# ---------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)


# ---------------------------------------------------------------------------
# Worker / WorkerPool  (generative model)
# ---------------------------------------------------------------------------
@dataclass
class Worker:
    gpu_id:    int
    device:    torch.device
    engine:    Engine
    tokenizer: object


class WorkerPool:
    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if device_type == "cuda" else 1
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(
        self,
        source: str,
        model_tag: Optional[str] = None,
        step: Optional[int] = None,
    ):
        print(f"Initialising generative worker pool with {self.num_gpus} device(s)...")
        if self.num_gpus > 1:
            assert device_type == "cuda", (
                "Multiple workers require CUDA. cpu/mps supports only 1 worker."
            )

        for gpu_id in range(self.num_gpus):
            dev = (
                torch.device(f"cuda:{gpu_id}")
                if device_type == "cuda"
                else torch.device(device_type)
            )
            label = f"GPU {gpu_id}" if device_type == "cuda" else device_type
            print(f"  Loading generative model on {label}...")

            mdl, tok, _ = load_model(
                source, dev, phase="eval", model_tag=model_tag, step=step
            )
            worker = Worker(gpu_id=gpu_id, device=dev, engine=Engine(mdl, tok), tokenizer=tok)
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} generative worker(s) ready!\n")

    async def acquire_worker(self) -> Worker:
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        await self.available_workers.put(worker)


# ---------------------------------------------------------------------------
# Pydantic models + validation
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    messages:    List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens:  Optional[int]   = None
    top_k:       Optional[int]   = None


def validate_chat_request(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required")

    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Max {MAX_MESSAGES_PER_REQUEST} per request",
        )

    total = 0
    for i, msg in enumerate(request.messages):
        if not msg.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")
        if len(msg.content) > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} exceeds {MAX_MESSAGE_LENGTH} characters",
            )
        total += len(msg.content)

    if total > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Conversation exceeds {MAX_TOTAL_CONVERSATION_LENGTH} characters",
        )

    for i, msg in enumerate(request.messages):
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role '{msg.role}'. Use 'user' or 'assistant'",
            )

    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"temperature must be in [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}]",
            )

    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be in [{MIN_TOP_K}, {MAX_TOP_K}]",
            )

    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be in [{MIN_MAX_TOKENS}, {MAX_MAX_TOKENS}]",
            )


# ---------------------------------------------------------------------------
# Safety guardrail  (runs in a background thread via asyncio.to_thread)
# ---------------------------------------------------------------------------
def run_safety_check(
    query: str,
    tokenizer,
    model: DistilBertForSequenceClassification,
    dev: torch.device,
) -> str:
    """
    Returns the predicted safety label in UPPER CASE.
    Runs synchronously — always call via asyncio.to_thread.
    """
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_id = logits.argmax().item()

    raw_label = model.config.id2label[predicted_id]
    return raw_label.upper().replace(" ", "_")   # normalise e.g. "self harm" → "SELF_HARM"


def classify_safety_label(label: str):
    """
    Returns one of: 'SELF_HARM', 'DANGEROUS', 'MISINFORMATION', 'SAFE'
    or 'UNKNOWN' (logged as a warning — allowed through by default).
    """
    if label in SELF_HARM_LABELS:
        return "SELF_HARM"
    if label in DANGEROUS_LABELS:
        return "DANGEROUS"
    if label in MISINFORMATION_LABELS:
        return "MISINFORMATION"
    if label in SAFE_LABELS:
        return "SAFE"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Lifespan  (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Safety guardrail
    logger.info(f"Loading Safety Guardrail ({args.safety_repo})...")
    safety_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    app.state.safety_tokenizer = AutoTokenizer.from_pretrained(args.safety_repo)
    app.state.safety_model = (
        DistilBertForSequenceClassification
        .from_pretrained(args.safety_repo)
        .to(safety_device)
    )
    app.state.safety_model.eval()
    app.state.safety_device = safety_device

    # *** IMPORTANT: check this log line to verify your label names ***
    id2label = app.state.safety_model.config.id2label
    logger.info(f"Safety model labels: {id2label}")
    logger.info("Safety Guardrail ready!")

    # 2. Generative worker pool
    logger.info("Loading nanochat generative model(s)...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(
        args.source, model_tag=args.model_tag, step=args.step
    )

    logger.info(f"Server ready at http://{args.host}:{args.port}")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Static routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';",
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    return FileResponse(os.path.join("nanochat", "logo.svg"), media_type="image/svg+xml")


# ---------------------------------------------------------------------------
# Health + stats endpoints  (were missing — now added)
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    worker_pool = getattr(app.state, "worker_pool", None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0,
        "safety_guardrail": getattr(app.state, "safety_model", None) is not None,
        "safety_repo": args.safety_repo,
    }


@app.get("/stats")
async def stats():
    wp: WorkerPool = app.state.worker_pool
    return {
        "total_workers":     len(wp.workers),
        "available_workers": wp.available_workers.qsize(),
        "busy_workers":      len(wp.workers) - wp.available_workers.qsize(),
        "workers": [{"gpu_id": w.gpu_id, "device": str(w.device)} for w in wp.workers],
    }


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------
async def generate_stream(
    worker: Worker,
    tokens: List[int],
    temperature=None,
    max_new_tokens=None,
    top_k=None,
) -> AsyncGenerator[str, None]:
    temperature    = temperature    if temperature    is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k          = top_k          if top_k          is not None else args.top_k

    tok_assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    tok_bos           = worker.tokenizer.get_bos_token_id()

    accumulated: List[int] = []
    last_clean = ""

    for token_column, _masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1),
    ):
        token = token_column[0]
        if token == tok_assistant_end or token == tok_bos:
            break

        accumulated.append(token)
        current_text = worker.tokenizer.decode(accumulated)

        if not current_text.endswith("\ufffd"):
            new_text = current_text[len(last_clean):]
            if new_text:
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"


async def safety_rejection_stream(message: str) -> AsyncGenerator[str, None]:
    yield f"data: {json.dumps({'token': message, 'gpu': 'guardrail'}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"


# ---------------------------------------------------------------------------
# Chat completions endpoint
# ---------------------------------------------------------------------------
@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    validate_chat_request(request)

    # ------------------------------------------------------------------
    # 1. Safety gate
    # ------------------------------------------------------------------
    last_user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), ""
    )

    if last_user_msg:
        raw_label = await asyncio.to_thread(
            run_safety_check,
            last_user_msg,
            app.state.safety_tokenizer,
            app.state.safety_model,
            app.state.safety_device,
        )
        category = classify_safety_label(raw_label)

        logger.info(
            f"🛡️  Safety check: raw=[{raw_label}] category=[{category}] "
            f"query='{last_user_msg[:60]}...'"
        )

        if category == "SELF_HARM":
            msg = (
                "🚨 System: I cannot provide advice on this topic. "
                "If you are in immediate distress, please call your local "
                "emergency number or a crisis hotline immediately."
            )
            return StreamingResponse(safety_rejection_stream(msg), media_type="text/event-stream")

        elif category == "DANGEROUS":
            msg = (
                "🚨 System: This request involves dangerous actions or materials. "
                "I cannot assist with this query. Please consult a human professional."
            )
            return StreamingResponse(safety_rejection_stream(msg), media_type="text/event-stream")

        elif category == "MISINFORMATION":
            msg = (
                "⚠️ System: The premise of this query contains known medical misinformation. "
                "Please consult a certified doctor for scientifically backed information."
            )
            return StreamingResponse(safety_rejection_stream(msg), media_type="text/event-stream")

        elif category == "UNKNOWN":
            # Label not in any known set — log it so you can add it above
            logger.warning(
                f"⚠️  Unknown safety label '{raw_label}' — allowing through. "
                "Add this label to the appropriate set at the top of the file."
            )
            # To block unknown labels instead, uncomment:
            # msg = "⚠️ System: This query was flagged by the safety system."
            # return StreamingResponse(safety_rejection_stream(msg), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # 2. Generative model (safe queries)
    # ------------------------------------------------------------------
    logger.info("=" * 20)
    for msg in request.messages:
        logger.info(f"[{msg.role.upper()}]: {msg.content}")
    logger.info("-" * 20)

    wp: WorkerPool = app.state.worker_pool
    worker = await wp.acquire_worker()

    try:
        tok_bos             = worker.tokenizer.get_bos_token_id()
        tok_user_start      = worker.tokenizer.encode_special("<|user_start|>")
        tok_user_end        = worker.tokenizer.encode_special("<|user_end|>")
        tok_assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        tok_assistant_end   = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens: List[int] = [tok_bos]
        for msg in request.messages:
            if msg.role == "user":
                conversation_tokens.append(tok_user_start)
                conversation_tokens.extend(worker.tokenizer.encode(msg.content))
                conversation_tokens.append(tok_user_end)
            elif msg.role == "assistant":
                conversation_tokens.append(tok_assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(msg.content))
                conversation_tokens.append(tok_assistant_end)
        conversation_tokens.append(tok_assistant_start)

        async def stream_and_release():
            response_tokens: List[str] = []
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k,
                ):
                    try:
                        data = json.loads(chunk.removeprefix("data: ").strip())
                        if "token" in data:
                            response_tokens.append(data["token"])
                    except json.JSONDecodeError:
                        pass
                    yield chunk
            finally:
                # Always release the worker — even on client disconnect
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {''.join(response_tokens)}")
                logger.info("=" * 20)
                await wp.release_worker(worker)

        return StreamingResponse(stream_and_release(), media_type="text/event-stream")

    except Exception as exc:
        await wp.release_worker(worker)
        raise exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print(f"\nStarting NanoChat Web Server")
    print(f"  Safety repo : {args.safety_repo}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Top-k       : {args.top_k}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Host:Port   : {args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)  