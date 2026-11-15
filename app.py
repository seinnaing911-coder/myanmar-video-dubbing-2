import asyncio
import hmac
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import edge_tts
import streamlit as st
from faster_whisper import WhisperModel
from google import genai
from pydub import AudioSegment

st.set_page_config(page_title="Myanmar Video AI Studio", page_icon="🎬", layout="centered")

VOICE = os.getenv("MYANMAR_VOICE", "my-MM-ThihaNeural")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default))).strip()
    except Exception:
        return str(os.getenv(name, default)).strip()


def check_login():
    expected = secret("APP_PASSWORD")
    if not expected:
        st.error("APP_PASSWORD မသတ်မှတ်ရသေးပါ။ Streamlit Secrets ထဲတွင် ထည့်ပါ။")
        st.stop()
    if st.session_state.get("authenticated"):
        return
    st.title("🔐 Private Myanmar Video Studio")
    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)
    if submitted:
        if hmac.compare_digest(password, expected):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Password မှားနေပါတယ်။")
    st.stop()


def run_command(command, label):
    p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(f"{label} မအောင်မြင်ပါ။\n{(p.stderr or p.stdout)[-4000:]}")
    return p.stdout.strip()


def media_duration(path):
    value = run_command(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        "Video ကြာချိန် စစ်ဆေးခြင်း",
    )
    return float(value)


def parse_json_response(raw: str):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    return json.loads(raw)


def test_gemini_key(api_key: str):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=GEMINI_MODEL, contents="Reply with exactly: GEMINI_OK")
    return (response.text or "").strip()


def transcribe_video(video_path, progress):
    progress(0.08, "မူရင်းအသံကို နားထောင်ပြီး စကားပြောချိန်များ ရှာနေပါသည်။")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, info = model.transcribe(video_path, task="transcribe", language=None, beam_size=5, vad_filter=True, vad_parameters={"min_silence_duration_ms": 450})
    found = []
    for seg in segments:
        text = (seg.text or "").strip()
        start, end = max(0.0, float(seg.start)), float(seg.end)
        if text and end > start + 0.12:
            found.append({"id": len(found) + 1, "start": start, "end": end, "source": text})
    if not found:
        raise RuntimeError("မူရင်းအသံထဲမှ စကားပြောသံ မတွေ့ပါ။ Video အသံကို စစ်ဆေးပါ။")
    return found, getattr(info, "language", "unknown"), getattr(info, "language_probability", 0.0)


def translate_segments(items, api_key, progress):
    client = genai.Client(api_key=api_key)
    translated = []
    batches = [items[i:i + 25] for i in range(0, len(items), 25)]
    for batch_no, batch in enumerate(batches, start=1):
        payload = [{"id": x["id"], "text": x["source"]} for x in batch]
        prompt = "Translate every item into natural conversational Burmese suitable for voice dubbing. Do not summarize, omit, or combine lines. Return ONLY valid JSON with id and myanmar fields.\n" + json.dumps(payload, ensure_ascii=False)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config={"response_mime_type": "application/json"})
        data = parse_json_response(response.text)
        by_id = {int(x["id"]): str(x["myanmar"]).strip() for x in data}
        for item in batch:
            my = by_id.get(item["id"], "")
            if not my:
                raise RuntimeError(f"စာကြောင်း {item['id']} ကို ဘာသာမပြန်နိုင်ပါ။")
            translated.append({**item, "myanmar": my})
        progress(0.12 + 0.15 * batch_no / len(batches), f"မြန်မာဘာသာပြန်နေပါသည် ({min(batch_no * 25, len(items))}/{len(items)})")
    return translated


def wait_for_gemini_file(client, uploaded):
    current = uploaded
    for _ in range(60):
        state = getattr(getattr(current, "state", None), "name", "ACTIVE")
        if state in ("ACTIVE", "SUCCEEDED", "READY"):
            return current
        if state in ("FAILED", "ERROR"):
            raise RuntimeError("Gemini သို့ Video upload လုပ်ရာတွင် မအောင်မြင်ပါ။")
        time.sleep(2)
        current = client.files.get(name=current.name)
    raise RuntimeError("Gemini Video processing အချိန်ကုန်သွားပါသည်။")


def create_recap_plan(video_path, api_key, progress, max_clips=10):
    client = genai.Client(api_key=api_key)
    progress(0.12, "Recap အတွက် Video ကို Gemini သို့ upload လုပ်နေပါသည်။")
    uploaded = wait_for_gemini_file(client, client.files.upload(file=video_path))
    prompt = f"""Analyze this video and create a short Burmese movie/video recap plan.
Select at most {max_clips} important non-overlapping scenes. Return ONLY a JSON array.
Each item must have exactly: start (number of seconds), end (number of seconds), narration (natural Burmese narration, 1-3 sentences), title (short Burmese title).
Use real timestamps, do not invent scenes, and keep narration concise for voice-over.
"""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=[uploaded, prompt], config={"response_mime_type": "application/json"})
    plan = parse_json_response(response.text)
    duration = media_duration(video_path)
    clean = []
    for i, item in enumerate(plan[:max_clips], start=1):
        start = max(0.0, min(duration, float(item["start"])))
        end = max(start + 0.2, min(duration, float(item["end"])))
        narration = str(item["narration"]).strip()
        if narration and end > start:
            clean.append({"id": i, "start": start, "end": end, "myanmar": narration, "title": str(item.get("title", "Recap"))})
    if not clean:
        raise RuntimeError("Gemini က အသုံးပြုနိုင်သော Recap scene မပြန်ပေးပါ။")
    progress(0.26, f"Recap scene {len(clean)} ခု ရွေးပြီးပါပြီ။")
    return clean


async def make_tts(text, out_path):
    await edge_tts.Communicate(text=text, voice=VOICE).save(out_path)


def make_video_piece(source, start, source_len, target_len, output):
    if source_len <= 0.04 or target_len <= 0.04:
        raise RuntimeError("Video အပိုင်းကြာချိန် မမှန်ပါ။")
    factor = target_len / source_len
    run_command(["ffmpeg", "-y", "-ss", f"{start:.5f}", "-i", source, "-t", f"{source_len:.5f}", "-an", "-vf", f"setpts={factor:.9f}*(PTS-STARTPTS),fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", output], "Video အပိုင်း အချိန်ညှိခြင်း")


def write_srt(items, output):
    def stamp(seconds):
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    with open(output, "w", encoding="utf-8") as f:
        for i, item in enumerate(items, start=1):
            f.write(f"{i}\n{stamp(item['output_start'])} --> {stamp(item['output_end'])}\n{item['myanmar']}\n\n")


def create_recap_video(video_path, items, work, progress, preserve_gaps=False):
    duration = media_duration(video_path)
    async def tts_all():
        paths = []
        for i, item in enumerate(items, start=1):
            p = os.path.join(work, f"recap_speech_{i:05d}.mp3")
            await make_tts(item["myanmar"], p)
            paths.append(p)
            progress(0.30 + 0.22 * i / len(items), f"မြန်မာအသံ ထုတ်နေပါသည် ({i}/{len(items)})")
        return paths
    speech_paths = asyncio.run(tts_all())
    pieces, timeline, cursor = [], AudioSegment.empty().set_frame_rate(24000).set_channels(1), 0.0
    output_items = []
    for i, (item, speech_path) in enumerate(zip(items, speech_paths), start=1):
        start = max(0.0, min(duration, item["start"]))
        end = max(start + 0.2, min(duration, item["end"]))
        speech = AudioSegment.from_file(speech_path).set_channels(1).set_frame_rate(24000)
        speech_len = max(0.2, len(speech) / 1000.0)
        if preserve_gaps and start > cursor + 0.02:
            gap = start - cursor
            gp = os.path.join(work, f"recap_gap_{i:05d}.mp4")
            make_video_piece(video_path, cursor, gap, gap, gp)
            pieces.append(gp)
            timeline += AudioSegment.silent(duration=round(gap * 1000), frame_rate=24000).set_channels(1)
        vp = os.path.join(work, f"recap_video_{i:05d}.mp4")
        make_video_piece(video_path, start, end - start, speech_len, vp)
        pieces.append(vp)
        out_start = len(timeline) / 1000.0
        timeline += speech
        out_end = len(timeline) / 1000.0
        output_items.append({**item, "output_start": out_start, "output_end": out_end})
        cursor = end
        progress(0.54 + 0.20 * i / len(items), f"Recap clip နဲ့ အသံကို အချိန်ကိုက်ညှိနေပါသည် ({i}/{len(items)})")
    if preserve_gaps and cursor < duration - 0.02:
        tail = duration - cursor
        tp = os.path.join(work, "recap_tail.mp4")
        make_video_piece(video_path, cursor, tail, tail, tp)
        pieces.append(tp)
        timeline += AudioSegment.silent(duration=round(tail * 1000), frame_rate=24000).set_channels(1)
    concat = os.path.join(work, "recap_concat.txt")
    with open(concat, "w", encoding="utf-8") as f:
        for p in pieces:
            f.write("file '" + p.replace("'", "'\\''") + "'\n")
    joined = os.path.join(work, "recap_joined.mp4")
    run_command(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat, "-c", "copy", "-an", "-movflags", "+faststart", joined], "Recap Video ပေါင်းစပ်ခြင်း")
    mp3 = os.path.join(work, "myanmar_recap_voice.mp3")
    timeline.export(mp3, format="mp3", bitrate="192k")
    return joined, mp3, output_items, len(timeline) / 1000.0


def mux_video(video, audio, output, duration, subtitle=None):
    command = ["ffmpeg", "-y", "-i", video, "-i", audio]
    if subtitle:
        command += ["-i", subtitle]
    command += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.3f}", "-movflags", "+faststart"]
    if subtitle:
        command += ["-map", "2:0", "-c:s", "mov_text", "-metadata:s:s:0", "language=mya"]
    command += [output]
    run_command(command, "MP4 ထုတ်လုပ်ခြင်း")


def download_video_url(url, work, progress):
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp မရှိသေးပါ။ requirements.txt ကို install လုပ်ပြီး ပြန်စမ်းပါ။") from exc
    output = os.path.join(work, "downloaded_video.%(ext)s")
    progress(0.03, "Video link မှ Video ကို download လုပ်နေပါသည်။")
    opts = {"outtmpl": output, "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best", "merge_output_format": "mp4", "noplaylist": True, "quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise RuntimeError(f"Video link download မအောင်မြင်ပါ: {exc}") from exc
    candidates = [p for p in Path(work).glob("downloaded_video.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if not candidates:
        raise RuntimeError("Download ပြီးတဲ့ Video file မတွေ့ပါ။ Link ကို စစ်ဆေးပါ။")
    return str(candidates[0])


check_login()
st.title("🎬 Myanmar Video AI Studio")
st.caption("Video file သို့မဟုတ် YouTube/TikTok link မှ မြန်မာ Recap နှင့် Dubbing ပြုလုပ်ပါ။")
with st.sidebar:
    st.subheader("AI ဆက်တင်")
    st.write(f"မြန်မာအသံ: {VOICE}")
    st.write(f"Whisper: {WHISPER_MODEL}")
    st.write(f"Gemini: {GEMINI_MODEL}")
    st.caption(f"Upload limit: {MAX_UPLOAD_MB} MB")
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

api_key = secret("GEMINI_API_KEY")
if not api_key:
    st.warning("Streamlit Secrets ထဲတွင် GEMINI_API_KEY ထည့်ရန် လိုအပ်ပါသည်။")
else:
    if st.button("🔑 Gemini API Key စမ်းမည်", use_container_width=True):
        try:
            st.success(f"Gemini API အလုပ်လုပ်ပါသည်: {test_gemini_key(api_key)}")
        except Exception as exc:
            st.error(f"Gemini API Key မအောင်မြင်ပါ: {exc}")

source_mode = st.radio("Video ရင်းမြစ်", ["Upload file", "Video link"], horizontal=True)
video_file = None
video_url = ""
if source_mode == "Upload file":
    video_file = st.file_uploader("မူရင်း Video ကို Upload လုပ်ပါ", type=["mp4", "mov", "mkv", "webm"])
else:
    video_url = st.text_input("YouTube / TikTok / direct video link", placeholder="https://...")

mode = st.radio("လုပ်ဆောင်ချက်", ["AI Movie Recap", "Auto Dubbing"], horizontal=True)
recap_limit = st.slider("Recap scene အများဆုံးအရေအတွက်", 3, 12, 8) if mode == "AI Movie Recap" else 0

if video_file or video_url.strip():
    if video_file and video_file.size / (1024 * 1024) > MAX_UPLOAD_MB:
        st.error(f"{MAX_UPLOAD_MB} MB ထက် မကြီးရပါ။")
    elif st.button("🚀 စတင်မည်", type="primary", use_container_width=True, disabled=not bool(api_key)):
        bar, status = st.progress(0), st.empty()
        def progress(v, msg):
            bar.progress(max(0, min(100, int(v * 100))))
            status.info(msg)
        try:
            with tempfile.TemporaryDirectory(prefix="myanmar_video_ai_") as work:
                if video_file:
                    source = os.path.join(work, "source" + Path(video_file.name).suffix.lower())
                    with open(source, "wb") as f:
                        f.write(video_file.getbuffer())
                    base_name = Path(video_file.name).stem
                else:
                    source = download_video_url(video_url.strip(), work, progress)
                    base_name = "video_recap"
                if mode == "AI Movie Recap":
                    plan = create_recap_plan(source, api_key, progress, recap_limit)
                else:
                    items, language, probability = transcribe_video(source, progress)
                    st.info(f"မူရင်းဘာသာစကား: {language} ({probability:.0%})")
                    plan = translate_segments(items, api_key, progress)
                joined, mp3, output_items, out_duration = create_recap_video(source, plan, work, progress, preserve_gaps=(mode == "Auto Dubbing"))
                mp4 = os.path.join(work, "final_myanmar_video.mp4")
                srt = os.path.join(work, "myanmar_subtitles.srt")
                write_srt(output_items, srt)
                progress(0.86, "နောက်ဆုံး MP4 ကို ထုတ်လုပ်နေပါသည်။")
                mux_video(joined, mp3, mp4, out_duration, subtitle=srt)
                st.session_state.result_mp4 = Path(mp4).read_bytes()
                st.session_state.result_mp3 = Path(mp3).read_bytes()
                st.session_state.result_srt = Path(srt).read_bytes()
                st.session_state.result_name = base_name
                st.session_state.recap_plan = output_items
                progress(1.0, "ပြီးပါပြီ။")
        except Exception as exc:
            st.error("လုပ်ဆောင်မှု မအောင်မြင်ပါ။")
            st.exception(exc)

if st.session_state.get("result_mp4"):
    st.divider()
    st.subheader("✅ ရလဒ်")
    st.video(st.session_state.result_mp4)
    st.audio(st.session_state.result_mp3, format="audio/mp3")
    base = st.session_state.get("result_name", "video")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Download MP4", st.session_state.result_mp4, file_name=f"{base}_myanmar.mp4", mime="video/mp4", use_container_width=True)
    with c2:
        st.download_button("⬇️ Download MP3", st.session_state.result_mp3, file_name=f"{base}_myanmar.mp3", mime="audio/mpeg", use_container_width=True)
    with c3:
        st.download_button("⬇️ Download SRT", st.session_state.result_srt, file_name=f"{base}_myanmar.srt", mime="application/x-subrip", use_container_width=True)
    with st.expander("Recap script / subtitle စာသား"):
        for item in st.session_state.get("recap_plan", []):
            st.write(f"**{item.get('title', 'Scene')}** — {item['myanmar']}")

st.divider()
st.caption("API key ကို code ထဲ မထည့်ပါနှင့်။ Streamlit Secrets တွင် GEMINI_API_KEY နှင့် APP_PASSWORD ကို သတ်မှတ်ပါ။")
