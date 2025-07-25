"""
Python client library for the R2‑D2 translator API.
=================================================

This module defines a ``R2TranslatorClient`` class that wraps the
REST and WebSocket endpoints exposed by the FastAPI server.  It
provides convenient methods for synthesising phrases to audio and
recognising audio files back to text.  Example usage::

    client = R2TranslatorClient(base_url="http://localhost:8000", api_key="secret-key")
    # Synchronous synthesis
    wav_bytes = client.to_r2("hello there")
    with open("hello.wav", "wb") as f:
        f.write(wav_bytes)
    # Recognition
    text = client.from_r2("hello.wav")
    print(text)

For streaming use cases the client exposes asynchronous methods
``stream_to_r2`` and ``stream_from_r2`` which use the ``websockets``
package.  These methods allow you to receive incremental audio
chunks or recognition results.

Note: This client depends on the external packages ``requests`` and
``websockets``.  Ensure they are installed in your environment.
"""

from __future__ import annotations

import base64
import json
import os
from typing import AsyncGenerator, Optional

import requests

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore


class R2TranslatorClient:
    """Client for interacting with the R2‑D2 translator API."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'x-api-key': api_key})

    def to_r2(self, phrase: str, *, version: Optional[str] = None, reverb: bool = False) -> bytes:
        """Synchronously synthesise a phrase to a WAV file.

        Args:
            phrase: Text to synthesise.
            version: Optional lexicon version.
            reverb: Whether to apply reverb.

        Returns:
            Bytes of a WAV file.
        """
        payload = {'phrase': phrase}
        params = {}
        if version:
            params['version'] = version
        if reverb:
            params['reverb'] = 'true'
        resp = self.session.post(f"{self.base_url}/to_r2", json=payload, params=params)
        resp.raise_for_status()
        return resp.content

    def from_r2(self, wav_path: str, *, version: Optional[str] = None, return_all: bool = False) -> dict:
        """Synchronously recognise the phrase encoded in a WAV file.

        Args:
            wav_path: Path to a WAV file.
            version: Optional lexicon version.
            return_all: If true, return second best and confidence.

        Returns:
            A dictionary with recognised text and optional metadata.
        """
        with open(wav_path, 'rb') as fh:
            files = {'file': ('audio.wav', fh, 'audio/wav')}
            params = {}
            if version:
                params['version'] = version
            if return_all:
                params['return_all'] = 'true'
            resp = self.session.post(f"{self.base_url}/from_r2", files=files, params=params)
        resp.raise_for_status()
        return resp.json()

    async def stream_to_r2(self, phrase: str, *, version: Optional[str] = None, reverb: bool = False) -> AsyncGenerator[bytes, None]:
        """Asynchronously stream synthesised audio chunks for a phrase.

        Yields raw PCM bytes of approximately 500 ms each.  Requires
        the ``websockets`` package.
        """
        if websockets is None:
            raise RuntimeError("The websockets package is required for streaming use")
        url = self.base_url.replace('http', 'ws') + '/ws/to_r2'
        async with websockets.connect(url) as ws:
            init_msg = {'phrase': phrase, 'version': version, 'reverb': reverb}
            await ws.send(json.dumps(init_msg))
            try:
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        yield msg
                    else:
                        break
            except websockets.ConnectionClosed:
                return

    async def stream_from_r2(self, audio_generator: AsyncGenerator[bytes, None], *, version: Optional[str] = None) -> AsyncGenerator[dict, None]:
        """Asynchronously stream audio frames for recognition.

        Args:
            audio_generator: An async generator yielding raw PCM frames
                (bytes) to send to the server.  Frames may be of
                arbitrary length.
            version: Optional lexicon version.

        Yields:
            Dictionaries with keys ``text``, ``second_best`` and
            ``confidence`` on each interim update.
        """
        if websockets is None:
            raise RuntimeError("The websockets package is required for streaming use")
        url = self.base_url.replace('http', 'ws') + '/ws/from_r2'
        async with websockets.connect(url) as ws:
            async for frame in audio_generator:
                await ws.send(frame)
                response = await ws.recv()
                if isinstance(response, str):
                    yield json.loads(response)