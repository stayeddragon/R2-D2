## R2‑D2 Translator Backend

This repository contains an end‑to‑end backend for translating human
phrases into R2‑D2 style beeps and vice versa. It comprises a
customisable lexicon, a synthesiser capable of generating tones with
vibrato and filtering, a simple recogniser based on Mel‑frequency
cepstral coefficients (MFCCs) and dynamic time warping (DTW), a
FastAPI web server exposing synthesis and recognition endpoints, unit
tests and dependency definitions.

### Repository structure
* **lexicon.json** – Defines the mapping from English phrases and
words to sequences of sound segments. Each segment specifies
frequency, duration, envelope, filter parameters, vibrato, noise
mixture and random perturbations. It also includes a fallback
mapping from A–Z to single segments used when spelling out
unknown phrases. The alphabet has been extended to cover digits
and several punctuation marks.

* **synth\.py** – Implements the **`R2Synth`** class for constructing
audio waveforms from the lexicon. It now supports optional
cross‑fading between segments to avoid abrupt transitions, caches
immutable segments for faster synthesis and allows the sine vs.
square wave mix to be tuned. A convenience function
**`generate_r2`** wraps this class for one‑off synthesis.

* **recognizer\.py** – Contains the **`Recognizer`** class which loads the
lexicon, precomputes reference features and uses MFCCs plus
dynamic time warping to map incoming audio back to phrases. It
now includes delta and delta‑delta coefficients, per‑utterance
normalisation and an adaptive threshold computed from inter‑phrase
distances. If no good match is found it falls back to spelling
using energy‑based segmentation.

* **server\.py** – A FastAPI application exposing multiple routes:
**`/to_r2`** (synthesis), **`/from_r2`** (recognition), **`/health`**
(health‑check) and **`/lexicon`** (list available phrases). Audio
is streamed as WAV files and recognised phrases are returned as
JSON.

* **tests/** – Pytest suites exercising both synthesis and
recognition on known and unknown phrases.

* **requirements.txt** – Python dependencies required to run the
project.

* **README\.md** – This document.

### Lexicon customisation
The lexicon is entirely data‑driven. To customise the translator:

1. Edit or extend **`lexicon.json.`** You may add new entries to the
**`phrases`** or **`common_words`** sections, or define additional
letters/symbols in the alphabet mapping. Each new entry should
follow the parameter schema described at the top of **`synth\.py`**.

2. Reload the server or instantiate new **`R2Synth`** and **`Recognizer`**
instances to pick up changes.

When adding common words or phrases you are free to include more than
the initial numbers provided – the synthesiser and recogniser will
automatically handle them. Unrecognised input will always fall back
to the spelled‑out mode.

### Running the API server
Install the dependencies listed in **`requirements.txt`**(for example
via **`pip install -r requirements.txt`**). Then run the server with
**`uvicorn`**:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```
You can adjust the **`--workers`** parameter to handle more concurrent
requests. Both endpoints are asynchronous so they release the
thread while writing files or reading uploads.

### Authentication
Endpoints that modify or retrieve audio (**`/to_r2`**, **`/from_r2`**)
require a simple API key. Supply it via the **`X‑API‑KEY`** HTTP
header. The default key is **`secret‑key`** but you should change
this in a production deployment. The **`/health`**, **`/lexicon`** and
**`/metrics`** endpoints do not require authentication.

### POST **`/to_r2`**
Send JSON containing a **`"phrase"`** field. The response is a WAV
stream. Example using **`curl:`** 

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"phrase": "thank you"}' \
     http://localhost:8000/to_r2 --output thank_you.wav
```
### POST **`/from_r2`**
Upload a WAV file using a form field named **`file`**. The response is a
JSON object with a **`"text"`** field containing the recognised phrase:

```bash
curl -X POST -F "file=@thank_you.wav" http://localhost:8000/from_r2
```
If the recogniser cannot confidently match the audio to a known
phrase the returned text will be of the form **`"<spelled: XYZ>"`**
showing the spelled‑out letters.

### Streaming with WebSockets
In addition to the REST endpoints the server exposes two WebSocket
routes for low‑latency streaming:

* **`/ws/from_r2`** – Clients can stream raw 16‑bit PCM audio frames
and receive interim recognition results after each chunk as JSON
messages containing the best match, the second best and a
confidence score.

* **`/ws/to_r2`** – Clients send a single JSON message containing a
phrase and optionally a lexicon version and reverb flag. The
server streams back 500 ms chunks of synthesised audio in raw
PCM form.

These WebSockets enable interactive, conversational applications
without waiting for full uploads or downloads.

### Metrics
The **`/metrics`** endpoint returns simple counters (total requests,
per‑route requests, cumulative latency) in JSON. This can be
integrated with your monitoring stack for basic observability.

### Integration with Web Audio API or mobile clients
For browser‑based applications you can use the Web Audio API to play
WAV files returned by **`/to_r2`** and to capture microphone input for
upload to **`/from_r2`**. A typical workflow is:

1. Send the text you wish to speak to **`/to_r2`** and receive a WAV
stream.

2. Use the Web Audio API’s **`AudioContext.decodeAudioData`** to decode
the WAV into an **`AudioBuffer`** and then play it via an
**`AudioBufferSourceNode`**.

3. To recognise speech, record audio in the browser (for example via
**`MediaRecorder`**), convert the resulting blob to a WAV file and
upload it to **`/from_r2`**.

On mobile platforms (iOS/Android) similar functionality can be
achieved using the respective audio APIs. Ensure that the recorded
audio is mono and that the sample rate is either 22.05 kHz or a
multiple thereof – the recogniser will internally resample if
required.

### Concurrency and performance considerations
Both synthesis and recognition are CPU‑bound operations due to signal
processing. For light workloads a single worker process is
sufficient. For high‑traffic deployments you may spawn multiple
workers via the **`--workers`** flag to **`uvicorn`** or run behind a
process manager such as Gunicorn with an ASGI worker class. The
recogniser caches MFCC features for all lexicon entries at startup to
minimise per‑request overhead.

### Advanced features
This project offers several optional enhancements beyond the core
REST API. These features improve recognition accuracy, scalability
and developer ergonomics.

### Neural embedding model
The **`embedding_model.py`** module defines a Siamese neural network
implemented in PyTorch which learns compact embeddings of MFCC
sequences. A training script, **`train_embedding.py`**, synthesises
examples from the lexicon and optimises the network using a
contrastive loss. The recogniser automatically falls back to a
simple mean‑MFCC embedding when PyTorch is unavailable.

### Background task queue
Long‑running jobs can be offloaded to Celery workers when the
environment variable **`USE_CELERY=1`** is set. In this mode the
**`/to_r2`** and **`/from_r2`** endpoints accept an **`async=true`** query
parameter. Instead of blocking the request, the server enqueues a
task and returns a **`task_id`**. You can poll **`/task/{id}`** to obtain
the result. See **`tasks.py`** and **`celeryconfig.py`** for details.

### WebSocket streaming
Two WebSocket endpoints enable low‑latency operation: **`/ws/from_r2`**
allows clients to stream PCM audio frames and receive interim
recognition results, while **`/ws/to_r2`** streams synthesised PCM
chunks for a phrase. These routes underpin truly interactive
applications.

### Lexicon hot reload and versioning
The synthesiser monitors the modification time of the lexicon file
and reloads its contents when it changes. Lexicon files can carry
version suffixes (e.g. **`lexicon_v1.json`**), selectable via the
**`version`** query parameter. The **`/admin`** dashboard lists all
available versions and provides a reload button to refresh the
running server without restarting.

### Containerisation and orchestration
The repository includes a **`Dockerfile`** and **`docker-compose.yml`** to
package the API together with a Redis broker (for Celery) and a
Prometheus instance (for metrics). The **`start.sh`** script builds
and starts all services. Prometheus is configured via
**`prometheus.yml`** to scrape the **`/metrics`** endpoint.

### Continuous integration
The GitHub Actions workflow in **`.github/workflows/ci.yml`** installs
dependencies, runs flake8 and black for linting, executes the test
suite and builds the Docker images. It also spins up the
docker‑compose stack and performs a basic health check, ensuring
confidence before merging changes.

### Client SDKs
To simplify integration you can use the provided client libraries:

* **`r2translator\.py`** – A Python wrapper around the REST and
WebSocket APIs built on **`requests`** and **`websockets`**. It offers
synchronous and asynchronous methods for synthesis and
recognition.

* **`r2translator.js`** – An ES6 JavaScript module exposing functions
to call the API from the browser or Node.js using the Fetch and
WebSocket APIs. It returns Blobs or JSON objects as
appropriate.

### Admin dashboard and observability
Navigate to **`/admin`** to view a simple dashboard summarising
requests, available lexicon versions and a reload button. The
**`/metrics`** endpoint exposes counters for basic observability and
Prometheus can scrape them at regular intervals.