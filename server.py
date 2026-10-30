"""iris-listen: speech to text on your own GPU, Qwen3-ASR-1.7B behind OpenAI's transcription API.

    POST /v1/audio/transcriptions   multipart form → {"text": ...}
        file             the clip: WAV, FLAC or OGG (anything libsndfile reads), any rate, mono or not
        language         optional, e.g. "en": forces that language; without it the model tells
        prompt           optional: names and terms the clip may hold, which the model leans towards
        response_format  "json" (default) or "text"
        model            accepted and ignored (OpenAI's clients send it)
    GET  /health                    "ok"

Every request needs `Authorization: Bearer <token>`, the token in --token-file
(install.sh makes one); without it the answer is 401. So it is safe on any
address, behind a proxy or not.

Run it with this folder's venv:  .venv/bin/python server.py --token-file <file>
It listens on 127.0.0.1 by default. Each --host adds an address; one that is
not up yet (a VPN's) is bound anyway and starts answering when it comes up.

The model writes down what was said, never a translation: told "en" for
Hungarian speech, it still writes Hungarian. It hears a whole clip at once, so a
clip over 28 s is cut at its quietest moments into pieces the model's
fixed-size cache holds. One clip at a time; others wait.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import secrets
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile
import uvicorn
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger("iris-listen")

RATE = 16000
MAX_BYTES = 25 * 1024 * 1024   # OpenAI's limit too
MAX_PROMPT = 2000              # characters; the prompt shares the model's cache with the audio
# Prompt plus answer, in tokens. The prompt is ~50 tokens, 8 per second of
# audio (a 28 s piece: ~275) and the vocabulary; the answer 3-4 per second.
# ~115 KB of VRAM per token.
CACHE_TOKENS = 2048
MAX_NEW_TOKENS = 384


class Speech:
    """Qwen3-ASR-1.7B on the GPU, through transformers.

    It is an audio encoder in front of a small language model, so the text is
    written one token at a time: about 15 ms per token on an RTX 5060 Ti, 0.3 s
    for a 7 s sentence. Those 15 ms need a fixed-size KV cache and torch.compile
    with CUDA graphs (plain generate() takes 25 ms a token). The cache never
    changes shape, so the decoder step is compiled once, at warmup, and no clip
    length recompiles it. CUDA graphs belong to the thread that made them, so
    every call runs on the one model thread (`worker`).
    """

    def __init__(self, path: Path) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor, CompileConfig, StaticCache
        from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

        self.torch = torch
        self.languages = set(LANGUAGE_CODE_TO_NAME)
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            path, dtype=torch.bfloat16).to("cuda").eval()
        self.cache = StaticCache(config=self.model.config.get_text_config(),
                                 max_cache_len=CACHE_TOKENS)
        self.compile = CompileConfig(fullgraph=False, dynamic=False, mode="reduce-overhead")

    def warmup(self) -> None:
        """Compile the decoder (~6 s), so the first clip doesn't."""
        rng = np.random.default_rng(0)
        for seconds in (1, 5):
            self.recognize(rng.normal(0, 0.01, RATE * seconds).astype(np.float32), None, None)

    def recognize(self, samples: np.ndarray, language: str | None, context: str | None) -> str:
        """One piece of at most 28 s at 16 kHz."""
        torch = self.torch
        inputs = self.processor.apply_transcription_request(
            audio=samples, language=language, prompt=context,
        ).to(self.model.device, self.model.dtype)
        prompt = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            self.cache.reset()
            out = self.model.generate(
                **inputs, do_sample=False, past_key_values=self.cache,
                compile_config=self.compile,
                max_new_tokens=min(MAX_NEW_TOKENS, CACHE_TOKENS - prompt))
        return self.processor.decode(out[:, prompt:], return_format="transcription_only")[0]


def split_at_pauses(samples: np.ndarray, max_seconds: float = 28.0,
                    search_seconds: float = 6.0, window: int = 480) -> list[np.ndarray]:
    """Cut 16 kHz audio into pieces of at most `max_seconds`, each cut at the
    quietest 30 ms in the last `search_seconds` before the limit, so it falls
    between words."""
    pieces, start, limit = [], 0, int(max_seconds * RATE)
    while len(samples) - start > limit:
        hi = start + limit
        lo = hi - int(search_seconds * RATE)
        frames = samples[lo:hi][: (hi - lo) // window * window].reshape(-1, window)
        cut = lo + int(np.argmin((frames ** 2).mean(axis=1))) * window + window // 2
        pieces.append(samples[start:cut])
        start = cut
    pieces.append(samples[start:])
    return pieces


def decode(data: bytes) -> np.ndarray:
    """The clip as 16 kHz mono float32."""
    samples, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if rate != RATE and len(samples):
        n = round(len(samples) * RATE / rate)
        samples = np.interp(np.linspace(0, len(samples) - 1, n),
                            np.arange(len(samples)), samples).astype(np.float32)
    return samples


token = b""                 # from --token-file, before the first request
speech: Speech | None = None
worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")
bearer = HTTPBearer(auto_error=False)


def authorized(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    if cred is None or not secrets.compare_digest(cred.credentials.encode(), token):
        raise HTTPException(401, "a bearer token is needed", headers={"WWW-Authenticate": "Bearer"})


# No /docs or /openapi.json: the API is in this docstring and the README.
app = FastAPI(dependencies=[Depends(authorized)], docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
def health():
    return PlainTextResponse("ok")


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile,
    language: str = Form(""),
    prompt: str = Form(""),
    response_format: str = Form("json"),
    model: str = Form(""),          # OpenAI's; accepted and ignored
):
    if response_format not in ("json", "text"):
        raise HTTPException(400, 'response_format: "json" or "text"')
    language = language.strip().lower() or None
    if language and language not in speech.languages:
        raise HTTPException(400, f"language: one of {', '.join(sorted(speech.languages))}")
    context = prompt.strip()[:MAX_PROMPT] or None
    if file.size is not None and file.size > MAX_BYTES:
        raise HTTPException(413, f"the clip is over {MAX_BYTES // (1024 * 1024)} MB")
    data = await file.read()
    try:
        samples = decode(data)
    except soundfile.LibsndfileError as exc:
        raise HTTPException(400, f"cannot read the clip: {exc}") from exc

    def recognize() -> str:
        t0 = time.time()
        texts = (speech.recognize(piece, language, context) or "" for piece in split_at_pauses(samples))
        text = " ".join(t.strip() for t in texts if t.strip())
        log.info("%.1fs audio -> %.2fs: %d chars", len(samples) / RATE, time.time() - t0, len(text))
        return text

    text = await asyncio.get_running_loop().run_in_executor(worker, recognize) if len(samples) else ""
    if response_format == "text":
        return PlainTextResponse(text)
    return JSONResponse({"text": text})


def main() -> None:
    global speech, token
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", action="append", help="address to listen on; repeat for more (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--model", type=Path, required=True, help="the folder holding Qwen3-ASR-1.7B's files")
    ap.add_argument("--token-file", required=True, type=Path,
                    help="file holding the bearer token every request must carry")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = args.token_file.expanduser().read_text().strip().encode()
    if not token:
        ap.error(f"{args.token_file} is empty")
    t0 = time.time()
    speech = worker.submit(Speech, args.model.expanduser()).result()
    worker.submit(speech.warmup).result()
    log.info("model loaded and warmed up in %.1fs", time.time() - t0)
    hosts = args.host or ["127.0.0.1"]
    log.info("ready on %s, port %d", ", ".join(hosts), args.port)
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[listen(h, args.port) for h in hosts])


def listen(host: str, port: int) -> socket.socket:
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # A VPN address may come up after this service; bind it now and serve it once it does.
    sock.setsockopt(socket.IPPROTO_IP, getattr(socket, "IP_FREEBIND", 15), 1)   # 15 on Linux
    sock.bind((host, port))
    return sock


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the model is a local folder; never ask the Hub
    main()
