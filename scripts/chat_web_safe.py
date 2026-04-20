#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.
Includes:
  - Data Parallelism for the heavy generative model
  - DistilBERT Safety Guardrail
  - Production-grade PII Masking Layer (Microsoft Presidio)

Launch:
  python -m scripts.chat_web_safe -n 2 --safety-repo anshul32467/Medical-DistilBert --host 0.0.0.0 --port 8090
"""

import argparse, json, os, torch, asyncio, logging, random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
from dataclasses import dataclass
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from transformers import AutoTokenizer, DistilBertForSequenceClassification

# ── PII layer ──────────────────────────────────────────────────────────────
from pii_layer import get_pii_masker, mask_pii_detailed, PIIMaskResult

# ---------------------------------------------------------------------------
# Abuse-prevention limits
# ---------------------------------------------------------------------------
MAX_MESSAGES_PER_REQUEST      = 500
MAX_MESSAGE_LENGTH            = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.0, 2.0
MIN_TOP_K,       MAX_TOP_K       = 0,   200
MIN_MAX_TOKENS,  MAX_MAX_TOKENS  = 1,   4096

# ---------------------------------------------------------------------------
# Safety label sets  (all matched in UPPER CASE)
# Model labels: {0:'SAFE', 1:'MEDICAL', 2:'DANGEROUS', 3:'SELF_HARM', 4:'MISINFORMATION'}
# ---------------------------------------------------------------------------
SELF_HARM_LABELS      = {"SELF_HARM", "SELF-HARM", "SELFHARM", "SUICIDE", "SELF_INJURY"}
DANGEROUS_LABELS      = {"DANGEROUS", "DANGER", "HARMFUL", "VIOLENCE", "TOXIC"}
MISINFORMATION_LABELS = {"MISINFORMATION", "MISINFO", "FAKE", "FALSE", "MISLEADING"}
SAFE_LABELS           = {"SAFE", "NORMAL", "BENIGN", "LEGITIMATE", "GENERAL", "MEDICAL",
                         "LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3", "LABEL_4"}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Primum AI, an empathetic, highly knowledgeable, and responsible medical AI assistant.

Your goal is to explain medical concepts, common treatments, and health information in a clear, easy-to-understand, and professional tone. Structure your answers logically so they are easy to read.

Important Guidelines:
- Be genuinely helpful and informative. Answer the user's questions thoroughly based on established medical science.
- Always be clear that you are an AI, not a doctor. Gently remind users to consult a licensed healthcare professional for official diagnosis, treatment, or medications.
- If a user describes a severe or life-threatening emergency, immediately advise them to contact emergency services (like 911).
- If you see placeholders like [PERSON] or [PHONE], treat them as if they are real names or numbers in the conversation.
- If you do not know the answer, politely admit it rather than guessing."""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="NanoChat Web Server with DistilBERT Safety Guardrail + PII Layer")
parser.add_argument("-n", "--num-gpus",    type=int,   default=1)
parser.add_argument("-i", "--source",      type=str,   default="sft")
parser.add_argument("-t", "--temperature", type=float, default=0.8)
parser.add_argument("-k", "--top-k",       type=int,   default=50)
parser.add_argument("-m", "--max-tokens",  type=int,   default=512)
parser.add_argument("-g", "--model-tag",   type=str,   default=None)
parser.add_argument("-s", "--step",        type=int,   default=None)
parser.add_argument("-p", "--port",        type=int,   default=8000)
parser.add_argument("--device-type",       type=str,   default="", choices=["cuda", "cpu", "mps"])
parser.add_argument("--host",              type=str,   default="0.0.0.0")
parser.add_argument("--safety-repo",       type=str,   default="anshul32467/Medical-DistilBert")
parser.add_argument("--disable-pii",       action="store_true", default=False,
                    help="Disable PII masking (use only in trusted dev environments)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Logging + device
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# ---------------------------------------------------------------------------
# Worker / WorkerPool
# ---------------------------------------------------------------------------
@dataclass
class Worker:
    gpu_id:    int
    device:    torch.device
    engine:    Engine
    tokenizer: object

class WorkerPool:
    def __init__(self, num_gpus=None):
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if device_type == "cuda" else 1
        self.num_gpus = num_gpus
        self.workers  = []
        self.available_workers = asyncio.Queue()

    async def initialize(self, source, model_tag=None, step=None):
        print(f"Initialising {self.num_gpus} worker(s)...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "Multiple workers require CUDA."
        for gpu_id in range(self.num_gpus):
            dev = torch.device(f"cuda:{gpu_id}") if device_type == "cuda" else torch.device(device_type)
            print(f"  Loading on {'GPU '+str(gpu_id) if device_type=='cuda' else device_type}...")
            mdl, tok, _ = load_model(source, dev, phase="eval", model_tag=model_tag, step=step)
            w = Worker(gpu_id=gpu_id, device=dev, engine=Engine(mdl, tok), tokenizer=tok)
            self.workers.append(w)
            await self.available_workers.put(w)
        print(f"All {self.num_gpus} worker(s) ready!")

    async def acquire_worker(self): return await self.available_workers.get()
    async def release_worker(self, w): await self.available_workers.put(w)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    messages:    List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens:  Optional[int]   = None
    top_k:       Optional[int]   = None

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_chat_request(request: ChatRequest) -> None:
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail="Too many messages")
    total = 0
    for i, msg in enumerate(request.messages):
        if not msg.content:
            raise HTTPException(status_code=400, detail=f"Message {i} empty")
        if len(msg.content) > MAX_MESSAGE_LENGTH:
            raise HTTPException(status_code=400, detail=f"Message {i} too long")
        total += len(msg.content)
    if total > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(status_code=400, detail="Conversation too long")
    for i, msg in enumerate(request.messages):
        if msg.role not in ("user", "assistant"):
            raise HTTPException(status_code=400, detail=f"Invalid role '{msg.role}'")
    if request.temperature is not None and not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise HTTPException(status_code=400, detail="Invalid temperature")
    if request.top_k is not None and not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
        raise HTTPException(status_code=400, detail="Invalid top_k")
    if request.max_tokens is not None and not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
        raise HTTPException(status_code=400, detail="Invalid max_tokens")

# ---------------------------------------------------------------------------
# PII masking helper
# ---------------------------------------------------------------------------
async def apply_pii_mask(text: str) -> tuple[str, PIIMaskResult]:
    """
    Runs PII masking in a thread (CPU-bound) so it doesn't block the event loop.
    Returns (masked_text, result_metadata).
    """
    if args.disable_pii:
        from pii_layer import PIIMaskResult
        return text, PIIMaskResult(original_length=len(text), masked_text=text)

    result: PIIMaskResult = await asyncio.to_thread(
        mask_pii_detailed, text
    )

    if result.pii_detected:
        logger.info(
            "PII masked | entities=%s count=%d time=%.1fms",
            result.entities_detected, result.entity_count, result.processing_time_ms,
        )
        if result.error:
            logger.warning("PII masking encountered an error: %s", result.error)

    # ------------------------------------------------------------------
    # Token / audit persistence — COMMENTED OUT.
    # Uncomment when you're ready to wire in a DB session:
    #
    # async with get_db_session() as session:
    #     await result.save_to_db(session)
    # ------------------------------------------------------------------

    return result.masked_text, result

# ---------------------------------------------------------------------------
# Safety guardrail
# ---------------------------------------------------------------------------
def run_safety_check(query: str, tokenizer, model, dev) -> str:
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        pid = model(**inputs).logits.argmax().item()
    return model.config.id2label[pid].upper().replace(" ", "_")

def classify_safety_label(label: str) -> str:
    if label in SELF_HARM_LABELS:      return "SELF_HARM"
    if label in DANGEROUS_LABELS:      return "DANGEROUS"
    if label in MISINFORMATION_LABELS: return "MISINFORMATION"
    if label in SAFE_LABELS:           return "SAFE"
    return "UNKNOWN"

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. PII layer (eager init so first request isn't slow)
    if not args.disable_pii:
        logger.info("Initialising PII masking layer...")
        pii = get_pii_masker()
        if pii._ready:
            logger.info("PII masking layer ready.")
        else:
            logger.warning("PII masking layer failed to initialise — PII will NOT be masked!")
        app.state.pii_masker = pii
    else:
        logger.warning("PII masking DISABLED by --disable-pii flag.")
        app.state.pii_masker = None

    # 2. Safety guardrail
    logger.info(f"Loading Safety Guardrail ({args.safety_repo})...")
    sd = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    app.state.safety_tokenizer = AutoTokenizer.from_pretrained(args.safety_repo)
    app.state.safety_model     = DistilBertForSequenceClassification.from_pretrained(args.safety_repo).to(sd)
    app.state.safety_model.eval()
    app.state.safety_device    = sd
    logger.info(f"Safety model labels: {app.state.safety_model.config.id2label}")
    logger.info("Safety Guardrail ready!")

    # 3. Generative model
    logger.info("Loading generative model(s)...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    logger.info(f"Server ready at http://{args.host}:{args.port}")
    yield

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# Static routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    with open(os.path.join("nanochat", "ui.html"), "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html)

@app.get("/logo.svg")
async def logo():
    return FileResponse(os.path.join("nanochat", "logo.svg"), media_type="image/svg+xml")

@app.get("/health")
async def health():
    wp  = getattr(app.state, "worker_pool", None)
    pii = getattr(app.state, "pii_masker",  None)
    return {
        "status":            "ok",
        "ready":             wp is not None and len(wp.workers) > 0,
        "num_gpus":          wp.num_gpus if wp else 0,
        "available_workers": wp.available_workers.qsize() if wp else 0,
        "safety_guardrail":  getattr(app.state, "safety_model", None) is not None,
        "safety_repo":       args.safety_repo,
        "pii_layer":         pii.health() if pii else {"ready": False, "disabled": True},
    }

@app.get("/stats")
async def stats():
    wp = app.state.worker_pool
    return {
        "total_workers":     len(wp.workers),
        "available_workers": wp.available_workers.qsize(),
        "busy_workers":      len(wp.workers) - wp.available_workers.qsize(),
        "workers":           [{"gpu_id": w.gpu_id, "device": str(w.device)} for w in wp.workers],
    }

# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------
async def generate_stream(worker, tokens, temperature=None, max_new_tokens=None, top_k=None):
    temperature    = temperature    if temperature    is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k          = top_k          if top_k          is not None else args.top_k
    tok_end = worker.tokenizer.encode_special("<|assistant_end|>")
    tok_bos = worker.tokenizer.get_bos_token_id()
    accumulated = []
    last_clean  = ""
    for token_column, _ in worker.engine.generate(
        tokens, num_samples=1, max_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, seed=random.randint(0, 2**31 - 1)
    ):
        token = token_column[0]
        if token == tok_end or token == tok_bos:
            break
        accumulated.append(token)
        current = worker.tokenizer.decode(accumulated)
        if not current.endswith("\ufffd"):
            new = current[len(last_clean):]
            if new:
                yield f"data: {json.dumps({'token': new, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean = current
    yield f"data: {json.dumps({'done': True})}\n\n"

async def safety_rejection_stream(message: str):
    yield f"data: {json.dumps({'token': message, 'gpu': 'guardrail'}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"

# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------
@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    validate_chat_request(request)

    # ── Step 1: PII masking — mask all messages before ANY further processing ──
    # We mask every message so PII never reaches the safety model OR the LLM.
    masked_messages = []
    for msg in request.messages:
        masked_content, _pii_result = await apply_pii_mask(msg.content)
        masked_messages.append(ChatMessage(role=msg.role, content=masked_content))

    # ── Step 2: Safety gate (runs on already-masked text) ──────────────────────
    last_user_msg = next(
        (m.content for m in reversed(masked_messages) if m.role == "user"), ""
    )
    if last_user_msg:
        raw = await asyncio.to_thread(
            run_safety_check, last_user_msg,
            app.state.safety_tokenizer, app.state.safety_model, app.state.safety_device
        )
        cat = classify_safety_label(raw)
        logger.info(
            "Shield Safety: raw=[%s] category=[%s] query='%s...'",
            raw, cat, last_user_msg[:60],
        )

        if cat == "SELF_HARM":
            return StreamingResponse(safety_rejection_stream(
                "I'm not able to respond to this message. "
                "If you are in crisis or having thoughts of harming yourself, "
                "please call emergency services or a crisis helpline immediately."
            ), media_type="text/event-stream")

        elif cat == "DANGEROUS":
            return StreamingResponse(safety_rejection_stream(
                "I'm not able to respond to this message. "
                "This request has been flagged as potentially dangerous and I cannot assist with it."
            ), media_type="text/event-stream")

        elif cat == "MISINFORMATION":
            return StreamingResponse(safety_rejection_stream(
                "I'm not able to respond to this message. "
                "This query appears to be based on medical misinformation. "
                "Please rely on verified medical sources and consult a certified doctor."
            ), media_type="text/event-stream")

        elif cat == "UNKNOWN":
            logger.warning("Unknown safety label '%s' — allowing through. Add to label sets if needed.", raw)

    # ── Step 3: Build token sequence (uses masked_messages, not originals) ──────
    logger.info("=" * 20)
    # Log roles only — content has already been PII-masked so this is safe.
    for msg in masked_messages:
        logger.info("[%s]: %s", msg.role.upper(), msg.content)
    logger.info("-" * 20)

    wp     = app.state.worker_pool
    worker = await wp.acquire_worker()

    try:
        tok_bos = worker.tokenizer.get_bos_token_id()
        tok_us  = worker.tokenizer.encode_special("<|user_start|>")
        tok_ue  = worker.tokenizer.encode_special("<|user_end|>")
        tok_as  = worker.tokenizer.encode_special("<|assistant_start|>")
        tok_ae  = worker.tokenizer.encode_special("<|assistant_end|>")

        conv = [tok_bos]

        # System prompt injection
        conv.append(tok_us)
        conv.extend(worker.tokenizer.encode(SYSTEM_PROMPT))
        conv.append(tok_ue)
        conv.append(tok_as)
        conv.extend(worker.tokenizer.encode(
            "Understood. I am Primum AI, a responsible medical AI assistant. "
            "I will follow all guidelines and provide only accurate, "
            "evidence-based medical information."
        ))
        conv.append(tok_ae)

        # Actual conversation — using PII-masked content
        for msg in masked_messages:
            if msg.role == "user":
                conv.append(tok_us)
                conv.extend(worker.tokenizer.encode(msg.content))
                conv.append(tok_ue)
            elif msg.role == "assistant":
                conv.append(tok_as)
                conv.extend(worker.tokenizer.encode(msg.content))
                conv.append(tok_ae)
        conv.append(tok_as)

        async def stream_and_release():
            resp = []
            try:
                async for chunk in generate_stream(
                    worker, conv,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k,
                ):
                    try:
                        d = json.loads(chunk.removeprefix("data: ").strip())
                        if "token" in d:
                            resp.append(d["token"])
                    except Exception:
                        pass
                    yield chunk
            finally:
                logger.info("[ASSISTANT] (GPU %s): %s", worker.gpu_id, "".join(resp))
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
    pii_status = "disabled" if args.disable_pii else "enabled"
    print(f"Starting NanoChat | safety: {args.safety_repo} | PII masking: {pii_status} | {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)