# Myanmar Auto Dubbing Studio

Upload one MP4/MOV/MKV/WEBM video. The app automatically detects speech language (intended for Chinese or English) with Whisper, translates segments to Burmese using Gemini, generates Burmese speech with Edge TTS, and retimes video segments to fit the natural TTS duration. It exports MP4 and MP3.

## Streamlit Community Cloud setup

1. Replace repository `app.py` with this version.
2. Replace `requirements.txt` with the supplied version.
3. Keep `packages.txt` containing `ffmpeg` and keep `.streamlit/config.toml`.
4. In Streamlit app Settings -> Secrets, keep your existing password and add your Gemini API key:

```toml
APP_PASSWORD = "your-existing-password"
GEMINI_API_KEY = "your-gemini-api-key"
```

Do not put API keys in app.py or commit them to GitHub.

## Notes

- SRT upload is no longer required.
- Whisper `base` multilingual model is downloaded on first run and uses CPU int8.
- The first run may be slow; Streamlit Community Cloud memory/CPU limits may constrain long videos.
- Translation requires a valid Gemini API key and may incur API charges.
- Automatic speech segmentation is approximate; overlapping speakers, music, noisy audio, and very fast speech can reduce accuracy.
