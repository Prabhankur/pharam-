from __future__ import annotations
import os
import subprocess
import tempfile
import base64
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="KisaanVaani 🌾")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Read API key — supports both spellings ───────────────────────────────────
SARVAM_API_KEY = (
    os.getenv("sarvamai_apy_key") or
    os.getenv("SARVAM_API_KEY") or
    os.getenv("sarvam_api_key") or
    ""
)
SARVAM_BASE = "https://api.sarvam.ai"

# ── Language config ──────────────────────────────────────────────────────────
LANG_MAP = {
    "hindi":   {"code": "hi-IN", "label": "हिंदी",   "speaker": "meera"},
    "punjabi": {"code": "pa-IN", "label": "ਪੰਜਾਬੀ", "speaker": "meera"},
    "tamil":   {"code": "ta-IN", "label": "தமிழ்",   "speaker": "arvind"},
}

# ── System prompts ───────────────────────────────────────────────────────────
FARMING_SYSTEM_PROMPT = """You are KisaanVaani (किसान वाणी), an expert agricultural advisor for Indian farmers.
You help with:
1. CROP ADVISORY – Best crops for the season, soil type, and region
2. PEST & DISEASE – Diagnose symptoms; suggest organic/chemical treatment
3. FERTILIZER – Dosage, timing, NPK recommendations
4. IRRIGATION – When/how much water based on crop and weather
5. WEATHER TIPS – What to do or avoid given weather conditions
6. MANDI PRICES – Approximate wholesale prices for common Indian crops
7. GOVT SCHEMES – PM Kisan, Fasal Bima, KCC, Soil Health Card, eNAM
8. CROP CALENDAR – Month-wise Kharif/Rabi/Zaid activities

RULES: Keep responses SHORT (max 3 sentences) — they will be read aloud.
Use simple everyday language. Be specific and actionable. Always be encouraging."""

IMAGE_SYSTEM_PROMPT = """You are KisaanVaani, an expert agricultural advisor specializing in
visual diagnosis of crops, soil, and fields. When given an image:
- Identify the crop (if visible)
- Diagnose any visible disease, pest damage, nutrient deficiency, or soil issue
- Suggest specific treatment (organic or chemical) with dosage
- Give 1-2 preventive tips for the future
Keep your answer concise (4-5 sentences max) since it will be spoken aloud."""

DOCUMENT_SYSTEM_PROMPT = """You are KisaanVaani, an expert in Indian government agricultural schemes and forms.
When given a document or form image:
- Identify what this document/form is
- Explain each field/section in simple language that a farmer can understand
- Tell the farmer EXACTLY what to write in each field (with examples)
- Mention which supporting documents they need to attach
- Warn about common mistakes farmers make when filling this form
Use simple Hindi-friendly English. Be step-by-step and very clear."""

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_headers():
    return {"api-subscription-key": SARVAM_API_KEY}

def convert_audio_to_wav(audio_bytes: bytes) -> bytes:
    """Try ffmpeg conversion; if ffmpeg missing (Windows), return raw bytes."""
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp:
        tmp.write(audio_bytes)
        in_path = tmp.name
    out_path = in_path + ".wav"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1",
             "-f", "wav", out_path, "-loglevel", "error"],
            capture_output=True,
        )
        if result.returncode == 0 and os.path.exists(out_path):
            return open(out_path, "rb").read()
        # ffmpeg ran but failed — return raw and let Sarvam handle it
        return audio_bytes
    except (FileNotFoundError, OSError):
        # ffmpeg not installed on this machine — send raw audio to Sarvam
        return audio_bytes
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)

async def translate(client: httpx.AsyncClient, text: str, src: str, tgt: str) -> str:
    resp = await client.post(
        f"{SARVAM_BASE}/translate",
        headers={**get_headers(), "Content-Type": "application/json"},
        json={"input": text, "source_language_code": src,
              "target_language_code": tgt, "model": "mayura:v1",
              "enable_preprocessing": True},
    )
    if resp.status_code == 200:
        return resp.json().get("translated_text", text)
    return text

async def llm_chat(client: httpx.AsyncClient, system: str, user_content) -> str:
    """user_content can be a string or a list (for vision)."""
    messages = [{"role": "system", "content": system}]
    if isinstance(user_content, str):
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_content})

    resp = await client.post(
        f"{SARVAM_BASE}/v1/chat/completions",
        headers={**get_headers(), "Content-Type": "application/json"},
        json={"model": "sarvam-105b-conversations", "messages": messages,
              "max_tokens": 300, "temperature": 0.7},
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"LLM error: {resp.text}")
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or \
              data["choices"][0]["message"].get("reasoning_content") or ""
    return content.strip()

async def tts(client: httpx.AsyncClient, text: str, lang_code: str, speaker: str) -> str | None:
    resp = await client.post(
        f"{SARVAM_BASE}/text-to-speech",
        headers={**get_headers(), "Content-Type": "application/json"},
        json={"inputs": [text[:500]], "target_language_code": lang_code,
              "speaker": speaker, "model": "bulbul:v1",
              "pitch": 0, "pace": 1.0, "loudness": 1.5,
              "speech_sample_rate": 8000, "enable_preprocessing": True},
    )
    if resp.status_code == 200:
        audios = resp.json().get("audios", [])
        return audios[0] if audios else None
    return None

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

@app.get("/health")
async def health():
    return {"status": "ok", "api_key_set": bool(SARVAM_API_KEY)}

# ── 1. Voice → Voice ─────────────────────────────────────────────────────────
@app.post("/process")
async def process_voice(
    audio: UploadFile = File(...),
    language: str = Form(default="hindi"),
):
    if not SARVAM_API_KEY:
        raise HTTPException(status_code=500, detail="sarvamai_apy_key not set in .env")

    lang = LANG_MAP.get(language, LANG_MAP["hindi"])
    lc   = lang["code"]
    raw  = await audio.read()

    try:
        wav = convert_audio_to_wav(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            # STT — Saaras v3 (send as webm if ffmpeg not available)
            audio_mime = "audio/wav" if wav != raw else "audio/webm"
            audio_name = "audio.wav" if wav != raw else "audio.webm"
            stt = await client.post(
                f"{SARVAM_BASE}/speech-to-text",
                headers=get_headers(),
                files={"file": (audio_name, wav, audio_mime)},
                data={"language_code": lc, "model": "saaras:v3", "with_timestamps": "false"},
            )
            if stt.status_code != 200:
                raise HTTPException(status_code=500, detail=f"STT error: {stt.text}")
            transcript = stt.json().get("transcript", "").strip()
            if not transcript:
                raise HTTPException(status_code=400, detail="Could not understand audio. Please speak clearly.")

            en_query    = await translate(client, transcript, lc, "en-IN")
            en_response = await llm_chat(client, FARMING_SYSTEM_PROMPT, en_query)
            final       = await translate(client, en_response, "en-IN", lc)
            audio_b64   = await tts(client, final, lc, lang["speaker"])

        return JSONResponse({
            "transcript": transcript, "english_query": en_query,
            "english_response": en_response, "response": final,
            "audio": audio_b64, "language": language,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 2. Text shortcut (chips) ─────────────────────────────────────────────────
@app.post("/process-text")
async def process_text(body: dict):
    if not SARVAM_API_KEY:
        raise HTTPException(status_code=500, detail="sarvamai_apy_key not set in .env")

    text     = body.get("text", "")
    language = body.get("language", "hindi")
    lang     = LANG_MAP.get(language, LANG_MAP["hindi"])
    lc       = lang["code"]

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            en_query    = await translate(client, text, lc, "en-IN")
            en_response = await llm_chat(client, FARMING_SYSTEM_PROMPT, en_query)
            final       = await translate(client, en_response, "en-IN", lc)
            audio_b64   = await tts(client, final, lc, lang["speaker"])
        return JSONResponse({"response": final, "english_response": en_response,
                             "audio": audio_b64, "language": language})
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── 3. Image upload — field / soil / crop photo ──────────────────────────────
@app.post("/upload-image")
async def upload_image(
    image: UploadFile = File(...),
    language: str = Form(default="hindi"),
    question: str = Form(default=""),
):
    if not SARVAM_API_KEY:
        raise HTTPException(status_code=500, detail="sarvamai_apy_key not set in .env")

    lang      = LANG_MAP.get(language, LANG_MAP["hindi"])
    lc        = lang["code"]
    img_bytes = await image.read()
    img_b64   = base64.b64encode(img_bytes).decode()
    mime      = image.content_type or "image/jpeg"

    # Build vision message
    user_text = question if question else "Analyze this image and give farming advice."
    vision_content = [
        {"type": "image_url",
         "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        {"type": "text", "text": user_text},
    ]

    async with httpx.AsyncClient(timeout=90) as client:
        # Translate farmer's question to English
        en_question = await translate(client, user_text, lc, "en-IN") if question else user_text

        vision_content[1]["text"] = en_question
        en_response = await llm_chat(client, IMAGE_SYSTEM_PROMPT, vision_content)
        final       = await translate(client, en_response, "en-IN", lc)
        audio_b64   = await tts(client, final, lc, lang["speaker"])

    return JSONResponse({
        "response": final, "english_response": en_response,
        "audio": audio_b64, "language": language,
        "image_preview": f"data:{mime};base64,{img_b64}",
    })

# ── 4. Document upload — govt forms / scheme docs ────────────────────────────
@app.post("/upload-document")
async def upload_document(
    document: UploadFile = File(...),
    language: str = Form(default="hindi"),
    question: str = Form(default=""),
):
    if not SARVAM_API_KEY:
        raise HTTPException(status_code=500, detail="sarvamai_apy_key not set in .env")

    lang     = LANG_MAP.get(language, LANG_MAP["hindi"])
    lc       = lang["code"]
    doc_bytes = await document.read()
    filename  = document.filename or ""
    mime      = document.content_type or "application/octet-stream"

    doc_b64 = base64.b64encode(doc_bytes).decode()

    async with httpx.AsyncClient(timeout=90) as client:
        # If it's a PDF, convert first page to image via ffmpeg/pdftoppm
        if filename.lower().endswith(".pdf") or "pdf" in mime:
            img_bytes = await pdf_first_page_to_image(doc_bytes)
            if img_bytes:
                doc_b64 = base64.b64encode(img_bytes).decode()
                mime    = "image/png"

        user_text   = question if question else "Please explain this document and how to fill it."
        en_question = await translate(client, user_text, lc, "en-IN") if question else user_text

        vision_content = [
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{doc_b64}"}},
            {"type": "text", "text": en_question},
        ]

        en_response = await llm_chat(client, DOCUMENT_SYSTEM_PROMPT, vision_content)
        final       = await translate(client, en_response, "en-IN", lc)
        audio_b64   = await tts(client, final, lc, lang["speaker"])

    return JSONResponse({
        "response": final, "english_response": en_response,
        "audio": audio_b64, "language": language,
        "doc_preview": f"data:{mime};base64,{doc_b64}",
    })


async def pdf_first_page_to_image(pdf_bytes: bytes) -> bytes | None:
    """Convert first page of PDF to PNG using pdftoppm (poppler)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name
    out_prefix = pdf_path + "_page"
    try:
        subprocess.run(
            ["pdftoppm", "-r", "150", "-png", "-f", "1", "-l", "1",
             pdf_path, out_prefix],
            check=True, capture_output=True,
        )
        # Find generated file
        for candidate in [out_prefix + "-1.png", out_prefix + "-01.png",
                           out_prefix + "-001.png"]:
            if os.path.exists(candidate):
                data = open(candidate, "rb").read()
                os.unlink(candidate)
                return data
    except Exception:
        pass
    finally:
        os.unlink(pdf_path)
    return None