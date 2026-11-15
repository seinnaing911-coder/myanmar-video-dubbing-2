import asyncio
import hmac
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import edge_tts
import streamlit as st
from faster_whisper import WhisperModel
from google import genai
from pydub import AudioSegment

st.set_page_config(page_title="Myanmar Auto Dubbing Studio", page_icon="🎬", layout="centered")
VOICE = "my-MM-ThihaNeural"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


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
    st.title("🔐 Private Dubbing Studio")
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
    value = run_command(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path], "Video ကြာချိန် စစ်ဆေးခြင်း")
    return float(value)


def transcribe_video(video_path, progress):
    progress(0.08, "မူရင်းအသံကို နားထောင်ပြီး တရုတ်/အင်္ဂလိပ် စကားပြောချိန်များ ရှာနေပါသည်။ ပထမအကြိမ်တွင် AI model download လုပ်နိုင်ပါသည်။")
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
    language = getattr(info, "language", "unknown")
    probability = getattr(info, "language_probability", 0.0)
    return found, language, probability


def translate_segments(items, api_key, progress):
    client = genai.Client(api_key=api_key)
    translated = []
    batches = [items[i:i + 25] for i in range(0, len(items), 25)]
    for batch_no, batch in enumerate(batches, start=1):
        payload = [{"id": x["id"], "text": x["source"]} for x in batch]
        prompt = """Translate every item into natural, conversational Burmese (Myanmar language) suitable for voice dubbing. The input may be Chinese or English. Keep the meaning, names and numbers. Do not summarize, omit, or combine lines. Return ONLY a valid JSON array. Each item must have exactly id and myanmar fields, preserving every input id. Example: [{\"id\":1,\"myanmar\":\"မြန်မာဘာသာပြန်\"}]\nINPUT:\n""" + json.dumps(payload, ensure_ascii=False)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config={"response_mime_type": "application/json"})
        raw = (response.text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        data = json.loads(raw)
        by_id = {int(x["id"]): str(x["myanmar"]).strip() for x in data}
        for item in batch:
            my = by_id.get(item["id"], "")
            if not my:
                raise RuntimeError(f"စာကြောင်း {item['id']} ကို ဘာသာမပြန်နိုင်ပါ။")
            translated.append({**item, "myanmar": my})
        progress(0.12 + 0.15 * batch_no / len(batches), f"မြန်မာဘာသာပြန်နေပါသည် ({min(batch_no * 25, len(items))}/{len(items)})")
    return translated


async def make_tts(text, out_path):
    await edge_tts.Communicate(text=text, voice=VOICE).save(out_path)


def make_video_piece(source, start, source_len, target_len, output):
    if source_len <= 0.04 or target_len <= 0.04:
        raise RuntimeError("Video အပိုင်းကြာချိန် မမှန်ပါ။")
    factor = target_len / source_len
    run_command(["ffmpeg", "-y", "-ss", f"{start:.5f}", "-i", source, "-t", f"{source_len:.5f}", "-an", "-vf", f"setpts={factor:.9f}*(PTS-STARTPTS),fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", output], "Video အပိုင်း အချိန်ညှိခြင်း")


def create_dubbed_video(video_path, items, work, progress):
    duration = media_duration(video_path)
    async def tts_all():
        paths = []
        for i, item in enumerate(items, start=1):
            p = os.path.join(work, f"speech_{i:05d}.mp3")
            await make_tts(item["myanmar"], p)
            paths.append(p)
            progress(0.28 + 0.30 * i / len(items), f"မြန်မာအသံ ထုတ်နေပါသည် ({i}/{len(items)})")
        return paths
    speech_paths = asyncio.run(tts_all())
    pieces, timeline, cursor = [], AudioSegment.empty().set_frame_rate(24000).set_channels(1), 0.0
    for i, (item, speech_path) in enumerate(zip(items, speech_paths), start=1):
        start = max(cursor, min(duration, item["start"]))
        end = max(start, min(duration, item["end"]))
        if end <= start + 0.04:
            continue
        if start > cursor + 0.02:
            gap = start - cursor
            gp = os.path.join(work, f"gap_{i:05d}.mp4")
            make_video_piece(video_path, cursor, gap, gap, gp)
            pieces.append(gp)
            timeline += AudioSegment.silent(duration=round(gap * 1000), frame_rate=24000).set_channels(1)
        speech = AudioSegment.from_file(speech_path).set_channels(1).set_frame_rate(24000)
        speech_len = len(speech) / 1000.0
        vp = os.path.join(work, f"speech_video_{i:05d}.mp4")
        make_video_piece(video_path, start, end - start, speech_len, vp)
        pieces.append(vp)
        timeline += speech
        cursor = end
        progress(0.60 + 0.20 * i / len(items), f"Video နဲ့ အသံကို အချိန်ကိုက်ညှိနေပါသည် ({i}/{len(items)})")
    if cursor < duration - 0.02:
        tail = duration - cursor
        tp = os.path.join(work, "tail.mp4")
        make_video_piece(video_path, cursor, tail, tail, tp)
        pieces.append(tp)
        timeline += AudioSegment.silent(duration=round(tail * 1000), frame_rate=24000).set_channels(1)
    if not pieces:
        raise RuntimeError("ပေါင်းစပ်ရန် Video အပိုင်း မရှိပါ။")
    concat = os.path.join(work, "concat.txt")
    with open(concat, "w", encoding="utf-8") as f:
        for p in pieces:
            f.write("file '" + p.replace("'", "'\\''") + "'\n")
    joined = os.path.join(work, "joined.mp4")
    run_command(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat, "-c", "copy", "-an", "-movflags", "+faststart", joined], "Video အပိုင်းများ ပေါင်းစပ်ခြင်း")
    mp3 = os.path.join(work, "myanmar_dub.mp3")
    timeline.export(mp3, format="mp3", bitrate="192k")
    return joined, mp3, len(timeline) / 1000.0


def mux_video(video, audio, output, duration):
    run_command(["ffmpeg", "-y", "-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.3f}", "-movflags", "+faststart", output], "MP4 ထုတ်လုပ်ခြင်း")


check_login()
st.title("🎬 Myanmar Auto Dubbing Studio")
st.caption("Video တစ်ဖိုင်တည်းတင်ပါ။ တရုတ်/အင်္ဂလိပ်အသံကို အလိုအလျောက်သိရှိပြီး မြန်မာဘာသာပြန်၊ အသံထုတ်၊ Video ကို Sync လုပ်ပေးပါမည်။")
with st.sidebar:
    st.subheader("အသံဆက်တင်")
    st.write(f"မြန်မာအသံ: {VOICE}")
    st.write(f"Speech recognition: Whisper {WHISPER_MODEL}")
    st.write(f"Translation: {GEMINI_MODEL}")
    st.caption(f"Upload limit: {MAX_UPLOAD_MB} MB")
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

api_key = secret("GEMINI_API_KEY")
if not api_key:
    st.warning("ပထမဆုံး Streamlit Secrets ထဲမှာ GEMINI_API_KEY ထည့်ရန် လိုအပ်ပါသည်။")
video_file = st.file_uploader("မူရင်း Video ကိုသာ Upload လုပ်ပါ", type=["mp4", "mov", "mkv", "webm"])
st.info("SRT ဖိုင် မလိုပါ။ မူရင်းအသံကို အလိုအလျောက် စာသားပြောင်းပြီး မြန်မာဘာသာပြန်ပါမည်။ မြန်မာအသံကို ပုံမှန်အမြန်နှုန်းထားပြီး အသံနဲ့ကိုက်အောင် Video အပိုင်းကိုသာ အနှေး/အမြန်ညှိပါမည်။")
if video_file:
    size_mb = video_file.size / (1024 * 1024)
    st.write(f"ရွေးထားသော Video: **{video_file.name}** ({size_mb:.1f} MB)")
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"{MAX_UPLOAD_MB} MB ထက် မကြီးရပါ။")
    elif st.button("🚀 AUTO DUBBING စတင်မည်", type="primary", use_container_width=True, disabled=not bool(api_key)):
        bar, status = st.progress(0), st.empty()
        def progress(v, msg):
            bar.progress(max(0, min(100, int(v * 100))))
            status.info(msg)
        try:
            with tempfile.TemporaryDirectory(prefix="auto_myanmar_dub_") as work:
                source = os.path.join(work, "source" + Path(video_file.name).suffix.lower())
                with open(source, "wb") as f:
                    f.write(video_file.getbuffer())
                progress(0.03, "Video ကို စစ်ဆေးနေပါသည်။")
                items, language, probability = transcribe_video(source, progress)
                if language not in ("zh", "en"):
                    st.warning(f"AI က မူရင်းဘာသာစကားကို '{language}' ဟု ခန့်မှန်းထားပါသည်။ တရုတ်/အင်္ဂလိပ်မဟုတ်ပါက ဘာသာပြန်မှားနိုင်ပါသည်။")
                progress(0.12, f"အသံမှ စာကြောင်း {len(items)} ကြောင်း ရရှိပါသည်။ ခန့်မှန်းဘာသာစကား: {language} ({probability:.0%})")
                items = translate_segments(items, api_key, progress)
                joined, mp3, out_duration = create_dubbed_video(source, items, work, progress)
                mp4 = os.path.join(work, "final_myanmar_dub.mp4")
                progress(0.84, "နောက်ဆုံး MP4 ဖိုင်ကို ထုတ်လုပ်နေပါသည်။")
                mux_video(joined, mp3, mp4, out_duration)
                st.session_state.result_mp4 = Path(mp4).read_bytes()
                st.session_state.result_mp3 = Path(mp3).read_bytes()
                st.session_state.result_name = Path(video_file.name).stem
                progress(1.0, "ပြီးပါပြီ။ MP4 နဲ့ MP3 ကို Download လုပ်နိုင်ပါပြီ။")
                st.success(f"ပြီးပါပြီ။ {len(items)} စာကြောင်းကို မြန်မာအသံဖြင့် ထုတ်လုပ်ပြီးပါပြီ။")
        except Exception as exc:
            st.error("လုပ်ဆောင်မှု မအောင်မြင်ပါ။")
            st.exception(exc)

if st.session_state.get("result_mp4"):
    st.divider()
    st.subheader("✅ ရလဒ်")
    st.video(st.session_state.result_mp4)
    st.audio(st.session_state.result_mp3, format="audio/mp3")
    base = st.session_state.get("result_name", "video")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Download MP4", st.session_state.result_mp4, file_name=f"{base}_myanmar_dub.mp4", mime="video/mp4", use_container_width=True)
    with c2:
        st.download_button("⬇️ Download MP3", st.session_state.result_mp3, file_name=f"{base}_myanmar_dub.mp3", mime="audio/mpeg", use_container_width=True)

st.divider()
st.caption("မှတ်ချက်: ပထမဆုံးအသုံးပြုချိန်တွင် Whisper AI model download လုပ်ရပါမည်။ Video ကြာချိန်နှင့် Server အရ processing အချိန်ကြာနိုင်ပါသည်။")
