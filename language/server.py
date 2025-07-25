"""
FastAPI server exposing R2‑D2 synthesis and recognition.
======================================================

This module defines a small API with two endpoints:

* ``POST /to_r2`` – Accepts a JSON object ``{"phrase": str}`` and
  returns a WAV file containing the synthesised R2‑D2 beeps for that
  phrase.  The ``Content-Disposition`` header hints the filename to
  clients.
* ``POST /from_r2`` – Accepts a file upload (``audio/wav``) and
  attempts to recognise the encoded phrase.  If successful the
  recognised phrase is returned; otherwise the spelled out form is
  returned.

The server instantiates a single ``R2Synth`` and ``Recognizer`` at
module load time.  This avoids reloading the lexicon and
recomputing reference features on every request.  Clients should
ensure that uploaded audio is sampled at 22.05 kHz or else the
recogniser will resample internally.

Example usage with ``curl``:

    curl -X POST -H "Content-Type: application/json" \
         -d '{"phrase": "hello"}' http://localhost:8000/to_r2 --output hello.wav

    curl -X POST -F "file=@hello.wav" http://localhost:8000/from_r2

Note on concurrency: FastAPI is asynchronous by default.  Both
endpoints are defined as ``async`` functions.  When deploying under
``uvicorn`` or another ASGI server you may increase the worker count
to handle multiple concurrent synthesis/recognition requests.

"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
import os
from typing import Dict, Optional, Any

import io
import base64
from scipy.io import wavfile
import numpy as np

from synth import R2Synth
from recognizer import Recognizer

# Try to import Celery tasks.  If Celery is not installed or the tasks
# module cannot be imported then asynchronous execution will be
# disabled.  Users may control asynchronous mode via the environment
# variable ``USE_CELERY``.
try:
    from tasks import synthesize_phrase as celery_synthesize_phrase
    from tasks import recognize_audio as celery_recognize_audio
    CELERY_AVAILABLE = True
except Exception:
    celery_synthesize_phrase = None  # type: ignore
    celery_recognize_audio = None  # type: ignore
    CELERY_AVAILABLE = False

# Determine whether to offload heavy work to Celery.  When set to a
# truthy value ("1", "true", etc.) the ``to_r2`` and ``from_r2``
# endpoints will accept an ``async" query parameter to trigger
# asynchronous execution.  Without Celery installed this flag is
# ignored.
USE_CELERY = CELERY_AVAILABLE and os.getenv('USE_CELERY', '0').lower() in ('1', 'true', 'yes')


app = FastAPI(title="R2D2 Translator", description="Synthesize and recognise R2‑D2 beeps", version="0.2")

# Instantiate synthesiser and recogniser once.  The recogniser loads
# the lexicon and precomputes features up front.
SYNTH = R2Synth()
# A cache of recognisers by lexicon version.  The default version
# corresponds to ``None`` and reuses the global synthesiser's sample rate.
RECOGNISERS: Dict[str | None, Recognizer] = {None: Recognizer(sample_rate=SYNTH.sample_rate)}

# Metrics storage for rudimentary observability.  These counters are
# incremented on each request.
METRICS = {
    'requests_total': 0,
    'to_r2_requests': 0,
    'from_r2_requests': 0,
    'latency_total': 0.0
}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return a JSON error.

    This handler ensures that unexpected errors result in a clear
    JSON response rather than an HTML traceback.  The status code
    defaults to 500.  You can augment this handler to log errors
    or translate specific exception types into custom codes.
    """
    # If the exception is already an HTTPException FastAPI will
    # handle it separately; we return a generic 500 response here.
    return JSONResponse(status_code=500, content={"detail": str(exc)})

# Simple API key for securing sensitive endpoints.  In production this
# should come from an environment variable or secret manager.
API_KEY = 'secret-key'


def _get_recognizer(version: str | None) -> Recognizer:
    """Get or create a recogniser for the specified lexicon version."""
    if version not in RECOGNISERS:
        # Compute lexicon path with version suffix
        if version:
            lex_path = os.path.join(os.path.dirname(__file__), f'lexicon_v{version}.json')
        else:
            lex_path = None
        RECOGNISERS[version] = Recognizer(lexicon_path=lex_path, sample_rate=SYNTH.sample_rate)
    return RECOGNISERS[version]


@app.post("/to_r2")
async def to_r2(payload: dict, request: Request) -> Any:
    """Generate R2‑D2 audio for a given phrase.

    This endpoint accepts a JSON body with a ``"phrase"`` field and
    optional query parameters ``version`` (lexicon version),
    ``reverb`` (apply reverb) and ``async``.  By default the audio
    is generated synchronously and streamed back as a WAV file.  If
    ``async=true`` and Celery is enabled, the request is enqueued
    and a JSON object containing the task ID is returned.  Clients
    can poll ``/task/{id}`` to obtain the result when it is ready.

    Raises:
        HTTPException: If the input JSON does not contain a phrase,
        the phrase is empty, or the API key is missing/invalid.
    """
    METRICS['requests_total'] += 1
    METRICS['to_r2_requests'] += 1
    # Validate payload
    if not isinstance(payload, dict) or 'phrase' not in payload:
        raise HTTPException(status_code=400, detail="JSON body must contain a 'phrase' field")
    phrase = str(payload.get('phrase', '')).strip()
    if not phrase:
        raise HTTPException(status_code=400, detail="Phrase must be non‑empty")
    # API key check
    key = request.headers.get('x-api-key')
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Parse query parameters
    params = request.query_params
    version: Optional[str] = params.get('version')
    reverb_flag: bool = params.get('reverb', 'false').lower() in ('1', 'true', 'yes')
    async_flag: bool = params.get('async', 'false').lower() in ('1', 'true', 'yes')
    # Asynchronous mode using Celery
    if async_flag and USE_CELERY:
        if not CELERY_AVAILABLE:
            raise HTTPException(status_code=503, detail="Celery is not available on the server")
        # Enqueue the synthesis task
        task = celery_synthesize_phrase.delay(phrase, version=version, reverb=reverb_flag)
        return JSONResponse({"task_id": task.id})
    # Synchronous mode
    try:
        synth = SYNTH if not version else R2Synth(version=version)
        audio, sr = synth.generate_r2(phrase, apply_reverb=reverb_flag)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during synthesis: {e}")
    pcm = (audio * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, pcm)
    buf.seek(0)
    filename = phrase.replace(' ', '_') + ".wav"
    return StreamingResponse(buf, media_type="audio/wav", headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.post("/from_r2")
async def from_r2(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """Recognise the phrase encoded in an uploaded WAV file.

    Clients upload a mono 16‑bit PCM WAV file under the form field
    named ``file``.  Optional query parameters ``version`` (lexicon
    version), ``return_all`` (return second best and confidence)
    and ``async`` (offload recognition to Celery) are supported.

    Returns:
        A JSON object containing the recognised text; optionally
        ``second_best`` and ``confidence`` fields.
    """
    METRICS['requests_total'] += 1
    METRICS['from_r2_requests'] += 1
    # API key check
    key = request.headers.get('x-api-key')
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if file.content_type not in ('audio/wav', 'audio/x-wav', 'audio/wave', 'audio/vnd.wave'):
        raise HTTPException(status_code=415, detail="Unsupported media type; only WAV files are accepted")
    data = await file.read()
    # Parse query parameters
    params = request.query_params
    version: Optional[str] = params.get('version')
    return_all: bool = params.get('return_all', 'false').lower() in ('1', 'true', 'yes')
    async_flag: bool = params.get('async', 'false').lower() in ('1', 'true', 'yes')
    # Asynchronous mode using Celery
    if async_flag and USE_CELERY:
        if not CELERY_AVAILABLE:
            raise HTTPException(status_code=503, detail="Celery is not available on the server")
        b64_data = base64.b64encode(data).decode('ascii')
        task = celery_recognize_audio.delay(b64_data, version=version, return_all=return_all)
        return JSONResponse({"task_id": task.id})
    # Synchronous mode
    try:
        sr, samples = wavfile.read(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio: {e}")
    # Convert to mono float
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.dtype.kind in 'iu':
        max_val = np.iinfo(samples.dtype).max
        audio = samples.astype(np.float32) / max_val
    elif samples.dtype.kind == 'f':
        audio = samples.astype(np.float32)
    else:
        raise HTTPException(status_code=400, detail="Unsupported sample format")
    recogniser = _get_recognizer(version) if version else RECOGNISERS[None]
    if return_all:
        best, second, conf = recogniser.recognize(audio, sr, return_all=True)
        return JSONResponse({"text": best, "second_best": second, "confidence": conf})
    else:
        phrase_out = recogniser.recognize(audio, sr, return_all=False)
        return JSONResponse({"text": phrase_out})

@app.get("/health")
async def health() -> JSONResponse:
    """Simple health‑check endpoint."""
    return JSONResponse({"status": "ok"})


@app.get("/lexicon")
async def lexicon() -> JSONResponse:
    """Return a summary of available phrases and common words.

    The keys are lower‑case strings corresponding to entries in the
    lexicon; the values are ignored.
    """
    return JSONResponse({
        "phrases": list(SYNTH.phrases.keys()),
        "common_words": list(SYNTH.common.keys())
    })


@app.get("/metrics")
async def metrics() -> JSONResponse:
    """Return collected metrics for basic observability."""
    return JSONResponse(METRICS)


# ---------------------------------------------------------------------
# Admin dashboard

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard() -> HTMLResponse:
    """Render a simple administrative dashboard.

    This page displays basic metrics, the currently loaded lexicon
    version and provides a button to reload the lexicon on the fly.
    In a production deployment you would secure this endpoint with
    authentication and serve a richer UI.
    """
    # Compute current lexicon version; assume None corresponds to base lexicon
    current_version = "default"
    # Determine if additional versions exist on disk
    base_dir = os.path.dirname(__file__)
    versions = ["default"]
    for fname in os.listdir(base_dir):
        if fname.startswith("lexicon_v") and fname.endswith(".json"):
            ver = fname[len("lexicon_v"):-len(".json")]
            versions.append(ver)
    metrics_html = "<ul>" + "".join([f"<li>{k}: {v}</li>" for k, v in METRICS.items()]) + "</ul>"
    versions_html = "<ul>" + "".join([f"<li>{v}</li>" for v in versions]) + "</ul>"
    html = f"""
    <html>
        <head><title>R2D2 Admin</title></head>
        <body>
            <h1>R2‑D2 Translator Admin Dashboard</h1>
            <h2>Metrics</h2>
            {metrics_html}
            <h2>Available lexicon versions</h2>
            {versions_html}
            <form method='post' action='/admin/reload'>
                <button type='submit'>Reload Lexicon</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/admin/reload")
async def admin_reload() -> JSONResponse:
    """Reload the lexicon files and refresh recognisers.

    This endpoint triggers the synthesiser to check whether the
    lexicon file has changed and reload its contents.  It also clears
    the recogniser cache so that future recognition requests use the
    updated lexicon.  Returns a JSON object indicating success.
    """
    # Trigger hot reload on the global synthesiser
    try:
        SYNTH._maybe_reload()
        # Clear recogniser cache so that new versions are instantiated
        RECOGNISERS.clear()
        # Recreate default recogniser
        RECOGNISERS[None] = Recognizer(sample_rate=SYNTH.sample_rate)
        return JSONResponse({"reloaded": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload lexicon: {e}")


# ---------------------------------------------------------------------
# Task status endpoint

@app.get("/task/{task_id}")
async def task_status(task_id: str) -> JSONResponse:
    """Check the status of an asynchronous synthesis or recognition task.

    This endpoint returns a JSON object containing the Celery task
    status (``PENDING``, ``STARTED``, ``SUCCESS``, ``FAILURE``) and
    result if available.  If Celery is disabled or the task ID is
    unknown, a 404 or 503 error is raised.

    Args:
        task_id: The identifier returned by ``/to_r2`` or ``/from_r2`` in
            asynchronous mode.

    Returns:
        A JSON object with keys ``status`` and optionally ``result``.
    """
    if not USE_CELERY or not CELERY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Celery is not available on the server")
    from celery.result import AsyncResult
    result = AsyncResult(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    response = {"status": result.status}
    if result.status == 'SUCCESS':
        # Try to parse JSON result; tasks return either base64 audio or dict
        try:
            response["result"] = result.get(timeout=0)
        except Exception:
            response["result"] = None
    elif result.status == 'FAILURE':
        response["result"] = str(result.result)
    return JSONResponse(response)


# ---------------------------------------------------------------------
# WebSocket endpoints for streaming recognition and synthesis
#

@app.websocket("/ws/from_r2")
async def websocket_from_r2(websocket: WebSocket):
    """Stream audio frames to the server and receive interim recognition results.

    The client should send raw 16‑bit PCM audio frames encoded as
    bytes.  After each chunk the server updates its buffer, decodes
    the full signal, performs recognition and sends back a JSON
    object containing the best match, the second best match and a
    confidence score.  This is a rudimentary implementation intended
    for demonstration; a production system would slice the signal
    into manageable windows and reset recognition state periodically.
    """
    await websocket.accept()
    buf = bytearray()
    try:
        while True:
            msg = await websocket.receive()
            if 'bytes' in msg:
                buf.extend(msg['bytes'])
                # Convert buffer to float32 audio
                # Interpret as little‑endian int16 samples
                samples = np.frombuffer(buf, dtype='<i2').astype(np.float32) / 32767.0
                if samples.size == 0:
                    continue
                # Use default recogniser for now (versioning can be added via query parameter if needed)
                recogniser = RECOGNISERS[None]
                best, second, conf = recogniser.recognize(samples, SYNTH.sample_rate, return_all=True)
                await websocket.send_json({"text": best, "second_best": second, "confidence": conf})
            elif 'text' in msg:
                # Ignore text frames
                continue
    except WebSocketDisconnect:
        return


@app.websocket("/ws/to_r2")
async def websocket_to_r2(websocket: WebSocket):
    """Stream synthesised audio chunks to the client.

    The client should send a single JSON message containing a
    ``"phrase"`` field.  The server synthesises the phrase and
    streams back raw 16‑bit PCM chunks of approximately 500 ms each.
    Once all chunks have been sent the connection is closed.
    """
    await websocket.accept()
    try:
        init_msg = await websocket.receive_json()
        phrase = init_msg.get('phrase')
        if not phrase:
            await websocket.close(code=1003)
            return
        version = init_msg.get('version')
        reverb = bool(init_msg.get('reverb'))
        synth = SYNTH if not version else R2Synth(version=version)
        audio, sr = synth.generate_r2(phrase, apply_reverb=reverb)
        pcm = (audio * 32767.0).astype(np.int16)
        chunk_samples = int(sr * 0.5)
        for i in range(0, len(pcm), chunk_samples):
            chunk = pcm[i:i + chunk_samples]
            await websocket.send_bytes(chunk.tobytes())
        await websocket.close()
    except WebSocketDisconnect:
        return