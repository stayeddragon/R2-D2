/*
 * JavaScript client library for the R2‑D2 translator API.
 *
 * This module exposes functions that call the REST and WebSocket
 * endpoints provided by the FastAPI server.  It can be used from
 * browser or Node.js contexts (fetch and WebSocket APIs must be
 * available).  Example usage:
 *
 *   import { toR2, fromR2, streamToR2, streamFromR2 } from './r2translator.js';
 *   const audioBlob = await toR2('hello', { apiKey: 'secret-key' });
 *   // Download audio
 *   const recognizerResult = await fromR2(fileInput.files[0], { apiKey: 'secret-key' });
 *
 */

/**
 * Synchronously synthesise a phrase to a WAV Blob.
 *
 * @param {string} phrase Text to synthesise
 * @param {Object} options Optional parameters: apiKey, version, reverb, baseUrl
 * @returns {Promise<Blob>} A Blob representing the WAV file
 */
export async function toR2(phrase, options = {}) {
  const baseUrl = options.baseUrl || 'http://localhost:8000';
  const url = new URL('/to_r2', baseUrl);
  if (options.version) url.searchParams.set('version', options.version);
  if (options.reverb) url.searchParams.set('reverb', 'true');
  const resp = await fetch(url.toString(), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': options.apiKey || 'secret-key'
    },
    body: JSON.stringify({ phrase })
  });
  if (!resp.ok) {
    throw new Error(`toR2 failed: ${resp.statusText}`);
  }
  const arrayBuffer = await resp.arrayBuffer();
  return new Blob([arrayBuffer], { type: 'audio/wav' });
}

/**
 * Synchronously recognise a WAV file and return the recognised text.
 *
 * @param {File|Blob} wavFile Audio file to upload
 * @param {Object} options Optional parameters: apiKey, version, returnAll, baseUrl
 * @returns {Promise<Object>} Recognition result
 */
export async function fromR2(wavFile, options = {}) {
  const baseUrl = options.baseUrl || 'http://localhost:8000';
  const url = new URL('/from_r2', baseUrl);
  if (options.version) url.searchParams.set('version', options.version);
  if (options.returnAll) url.searchParams.set('return_all', 'true');
  const formData = new FormData();
  formData.append('file', wavFile, 'audio.wav');
  const resp = await fetch(url.toString(), {
    method: 'POST',
    headers: {
      'x-api-key': options.apiKey || 'secret-key'
    },
    body: formData
  });
  if (!resp.ok) {
    throw new Error(`fromR2 failed: ${resp.statusText}`);
  }
  return await resp.json();
}

/**
 * Asynchronously stream synthesised audio for a phrase.
 *
 * @param {string} phrase Text to synthesise
 * @param {Object} options Optional parameters: version, reverb, baseUrl
 * @returns {WebSocket} A WebSocket that emits binary messages (ArrayBuffer)
 */
export function streamToR2(phrase, options = {}) {
  const baseUrl = options.baseUrl || 'ws://localhost:8000';
  const url = new URL('/ws/to_r2', baseUrl);
  const ws = new WebSocket(url.toString());
  ws.addEventListener('open', () => {
    ws.send(JSON.stringify({ phrase, version: options.version, reverb: !!options.reverb }));
  });
  return ws;
}

/**
 * Asynchronously stream audio frames to the recogniser.
 *
 * @param {function(ArrayBuffer):void} onResult Callback invoked with recognition result objects
 * @param {Object} options Optional parameters: version, baseUrl
 * @returns {WebSocket} A WebSocket to which raw PCM frames can be sent
 */
export function streamFromR2(onResult, options = {}) {
  const baseUrl = options.baseUrl || 'ws://localhost:8000';
  const url = new URL('/ws/from_r2', baseUrl);
  const ws = new WebSocket(url.toString());
  ws.addEventListener('message', (event) => {
    if (typeof event.data === 'string') {
      try {
        const obj = JSON.parse(event.data);
        onResult(obj);
      } catch (e) {
        console.error('streamFromR2: failed to parse message', e);
      }
    }
  });
  return ws;
}