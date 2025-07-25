## R2‑D2 Translator Backend: User Guide

This guide explains how to install, configure and use the R2‑D2 translator backend. The system converts human text into R2‑D2‑style beeps and recognises those beeps back to text. It consists of a customisable lexicon, a synthesis engine, a recogniser, a neural embedding model, a FastAPI server, background tasks via Celery, client SDKs and containerisation tooling.

### 1. Installation

**Prerequisites**

* Python 3.9 or newer (3.10+ recommended).
* **`pip`** package manager.
* Optionally, Redis if you want to enable asynchronous background tasks, and PyTorch if you plan to train/use the neural embedding model.

**Setup**

1. Clone your repository locally.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The requirements file installs NumPy, SciPy, FastAPI/Uvicorn, Celery, Redis clients and other essentials. To use the neural embedding model, install PyTorch separately:

   ```bash
   pip install torch
   ```

3. If you intend to use background tasks, run a Redis server locally or via Docker (see §4).

### 2. Running the API server

**Local development**

Launch the FastAPI application with Uvicorn:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

The server reads the lexicon from `lexicon.json`, exposes several REST endpoints, and requires an API key for synthesis and recognition endpoints. The default key is `secret-key`; override it by setting the `API_KEY` environment variable.

**Key endpoints**

* **POST `/to_r2`** – Synthesize beeps from text. Send JSON like `{ "phrase": "hello world" }` with header `X‑API‑KEY`. Optional query parameters:

  * `version` – use an alternate lexicon file (e.g. `version=1` for `lexicon_v1.json`).
  * `reverb=true` – apply a simple reverb effect.
  * `async=true` – queue the job via Celery (see §3).

  Example:

  ```bash
  curl -X POST -H "Content-Type: application/json" \
       -H "X-API-KEY: secret-key" \
       -d '{"phrase": "thank you"}' \
       "http://localhost:8000/to_r2?reverb=true" --output thank_you.wav
  ```

* **POST `/from_r2`** – Recognise beeps back to text. Upload a WAV file via multipart form (`file` field) and include `X‑API‑KEY`. Optional query parameters:

  * `version` – lexicon version.
  * `return_all=true` – include the second-best match and confidence score.
  * `async=true` – queue the job via Celery.

  Example:

  ```bash
  curl -X POST -H "X-API-KEY: secret-key" \
       -F "file=@thank_you.wav" \
       "http://localhost:8000/from_r2?return_all=true"
  ```

* **WebSocket `/ws/to_r2`** – Stream synthesised PCM audio chunks (\~500 ms each). Connect via WebSocket, send a JSON message like `{ "phrase": "hello", "reverb": false }`, then receive binary frames containing 16‑bit PCM data.

* **WebSocket `/ws/from_r2`** – Stream microphone audio frames for recognition. Send raw PCM bytes and receive JSON with the best match, second-best and confidence score after each chunk.

* **GET `/health`** – Returns `{"status": "ok"}`.

* **GET `/lexicon`** – Lists all phrases and common words in the current lexicon.

* **GET `/metrics`** – Returns simple counters (request counts and total latency) in JSON. Prometheus can scrape this endpoint.

* **GET `/admin`** – A basic HTML dashboard showing metrics, available lexicon versions and a “Reload Lexicon” button.

* **POST `/admin/reload`** – Reload the lexicon without restarting the server.

* **GET `/task/{task_id}`** – When using `async=true`, check the status of a Celery job. Returns the task status and result (if complete).

**Lexicon management**

Lexicons live in `lexicon.json` (default) and `lexicon_v*.json`. You can add new phrases, words or characters by editing these files. The synthesiser lower‑cases phrases and words and upper‑cases letters automatically. To hot‑reload changes, either restart the server or click the reload button on `/admin`. For A/B testing, copy the file with a new version suffix (`lexicon_v2.json`) and specify `version=2` in your requests.

### 3. Background tasks with Celery

To offload heavy synthesis or recognition jobs:

1. Ensure Redis is running (locally or via Docker).
2. Set `USE_CELERY=1` in the environment before starting the API.
3. Pass `async=true` on `/to_r2` or `/from_r2`.

The endpoint immediately returns `{ "task_id": "<id>" }`. Poll `/task/<id>` until the status becomes `SUCCESS` to retrieve the result. Celery tasks are defined in `tasks.py` and configuration in `celeryconfig.py`.

### 4. Containerisation and orchestration

A `Dockerfile` and `docker-compose.yml` package the API, Celery worker, Redis and Prometheus.

* Build and run everything:

  ```bash
  ./start.sh
  ```

  This builds images and starts `api` (FastAPI), `worker` (Celery), `redis`, and `prometheus`. The API listens on port 8000; Prometheus on 9090.

* Shut down the stack:

  ```bash
  docker-compose down
  ```

Prometheus scrapes metrics from the API as defined in `prometheus.yml`.

### 5. Neural embedding model

The recogniser uses MFCC+DTW by default. You can train a Siamese neural network to produce richer embeddings:

1. Install PyTorch: `pip install torch`.

2. Run the training script:

   ```bash
   python train_embedding.py --epochs 10 --lr 0.001 --output embedding.pth
   ```

   This synthesises training examples from the lexicon and trains the network using a contrastive loss.

3. Load and use the model with:

   ```python
   from embedding_model import load_model, compute_embedding
   model = load_model('embedding.pth')
   embedding = compute_embedding(audio_array, sample_rate, model)
   ```

Integration of the neural model into the recogniser is left as an exercise; the current recogniser computes a simple mean‑MFCC embedding for confidence scoring.

### 6. Client SDKs

**Python SDK (`r2translator.py`)**

```python
from r2translator import R2TranslatorClient

client = R2TranslatorClient(base_url="http://localhost:8000", api_key="secret-key")

# Synchronous synthesis
wav_bytes = client.to_r2("R2‑D2, where are you?", version="1", reverb=True)
with open("output.wav", "wb") as f:
    f.write(wav_bytes)

# Synchronous recognition
result = client.from_r2("output.wav", return_all=True)
print(result)  # {'text': 'r2‑d2, where are you?', 'second_best': ...}
```

Async streaming methods `stream_to_r2()` and `stream_from_r2()` use the `websockets` library to send/receive data incrementally.

**JavaScript SDK (`r2translator.js`)**

```js
import { toR2, fromR2, streamToR2, streamFromR2 } from './r2translator.js';

// Synchronous synthesis
const blob = await toR2('hello there', { apiKey: 'secret-key', version: '1', reverb: true });
const url = URL.createObjectURL(blob);
audio.src = url;

// Recognition
const result = await fromR2(fileInput.files[0], { apiKey: 'secret-key', returnAll: true });
console.log(result);

// Streaming synthesis
const wsOut = streamToR2('how are you?', { reverb: false });
wsOut.onmessage = (event) => {
  const pcmData = new Int16Array(event.data);
  // play pcmData...
};

// Streaming recognition
const wsIn = streamFromR2((msg) => console.log(msg));
// feed PCM frames to wsIn.send(...)
```

### 7. Testing

Install `pytest` and run:

```bash
pytest -q
```

The tests cover synthesis of known phrases, fallback spelling, recognition accuracy, resampling support and API behaviour with/without API keys.

### 8. Further customisation

* **Lexicon expansion:** Add more phrases, words or symbols to the lexicon JSON files. Unknown input will always fall back to spelling.
* **Additional effects:** Extend `R2Synth` with new waveform types, envelopes or post‑processing (e.g. chorus, distortion).
* **Improved recognition:** Integrate the neural model into `Recognizer` to replace or augment DTW matching.
* **Deployment:** Write Helm charts or Terraform scripts to deploy the Dockerised stack to Kubernetes or cloud platforms.


---

## 9. Component overview and purpose

To make the system easier to understand, here’s a plain‑language description of each major part of the R2‑D2 translator and its role in the overall architecture.

### Lexicon (`lexicon.json`, `lexicon_v*.json`)

The lexicon is the “vocabulary” of your translator. It maps English phrases, common words and individual characters (A–Z, digits and punctuation) to arrays of sound instructions—frequency, duration, envelope shape, filters, vibrato, noise mix and random shifts. When you ask the synthesiser to speak a phrase, it looks up the phrase (or falls back to spelling it letter by letter) and uses these parameters to generate R2‑D2 beeps. Because it’s just a JSON file, you can customise the lexicon without touching any code.

### Synthesiser (`R2Synth` in `synth.py`)

`R2Synth` is responsible for actually creating sounds. Given a sequence of lexicon entries, it uses NumPy and SciPy to generate sine and square waves, apply envelopes and vibrato, mix in noise, run the sound through digital filters and optionally add reverb. It concatenates these segments with optional cross‑fades to produce smooth output. The synthesiser also caches immutable segments for efficiency and monitors the lexicon file for changes so it can hot‑reload updates.

### Recogniser (`Recognizer` in `recognizer.py`)

The recogniser does the reverse: it takes an audio signal and tries to decide which lexicon entry produced it. It breaks the waveform into short frames, computes Mel‑frequency cepstral coefficients (MFCCs) and their first and second derivatives (delta and delta‑delta), and then compares the resulting sequence against precomputed templates for each phrase. Dynamic Time Warping (DTW) aligns sequences of different lengths, and a simple “embedding” (the average MFCC vector) provides a rough fingerprint. If no phrase is close enough, the recogniser falls back to splitting the audio into equal segments, matching each segment against letter templates and returning a spelled‑out string like `<spelled: R2D2>`.

### Neural embedding model (`embedding_model.py` and `train_embedding.py`)

The neural embedding model is an optional machine‑learning enhancement. Instead of comparing MFCC sequences directly, you can train a Siamese neural network to learn a compact vector representation (embedding) of each audio phrase. The network is trained so that embeddings of the same phrase are close together while embeddings of different phrases are far apart. During recognition, you compute the embedding of an incoming audio clip and find the nearest stored embedding. This approach can be more robust to noise and distortion than plain DTW. The provided `train_embedding.py` script generates synthetic training data from the lexicon and trains the model using contrastive loss. PyTorch is required for this component, but the rest of the system works without it.

### Celery (`celeryconfig.py` and `tasks.py`)

Celery is a distributed task queue. Heavy operations such as synthesising very long phrases or computing DTW for large audio files can block the web server for several seconds. By enabling Celery (`USE_CELERY=1` and running a Redis broker), you can offload those jobs to a background worker process. When a request includes `async=true`, the server places the task in the queue and immediately returns a task ID. The client polls `/task/<id>` to obtain the result when the worker has finished. This keeps your API responsive under load.

### FastAPI server (`server.py`)

The FastAPI application ties everything together and exposes a clean interface to clients. It defines REST endpoints for one‑shot synthesis and recognition, WebSocket endpoints for real‑time streaming, an admin dashboard, metrics and health checks. It also integrates optional Celery task dispatch and versioned lexicon selection. FastAPI automatically generates an OpenAPI/Swagger specification for developers to explore.

### Docker and Docker Compose

Containerisation simplifies deployment and local testing. The `Dockerfile` builds an image containing your Python code and dependencies. `docker-compose.yml` defines a multi‑service stack: the API (`api`), a Celery worker (`worker`), a Redis broker (`redis`) and Prometheus for metrics. Running `./start.sh` builds and launches all services in one command. This structure makes it easy to deploy the backend on your own machine, in the cloud or under an orchestrator like Kubernetes.

### Prometheus and metrics (`prometheus.yml` and `/metrics`)

Observability is vital in production. The API keeps simple counters (total requests, per‑endpoint counts, cumulative latency) and exposes them at `/metrics`. Prometheus can scrape these numbers at regular intervals (configured via `prometheus.yml`) and store them for graphing and alerting. The admin dashboard also displays the current metrics in a human‑readable form.

### Client SDKs (`r2translator.py` and `r2translator.js`)

To make it easy to consume the API, two lightweight client libraries are provided. The Python SDK wraps the REST and WebSocket endpoints using `requests` and `websockets`, offering synchronous and asynchronous methods. The JavaScript SDK uses the Fetch and WebSocket APIs and is suitable for browser or Node.js environments. These libraries abstract away HTTP details so you can focus on your application logic.

### Tests (`tests/`)

Unit tests verify that synthesis and recognition behave as expected, including edge cases like unknown phrases and resampling. They also exercise the API endpoints (with and without API keys) to ensure correct status codes and responses. Running `pytest` regularly helps catch regressions as you modify the code.

---

