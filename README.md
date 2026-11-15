# Myanmar Video AI Studio

Streamlit app for Burmese video dubbing and AI movie/video recap. It supports local upload or a YouTube/TikTok/direct video URL.

## Features

- Gemini API key connection test
- AI Movie Recap: Gemini analyzes the video, chooses important scenes, writes Burmese narration, and returns MP4, MP3, and SRT subtitles
- Auto Dubbing: Whisper detects speech, Gemini translates it to Burmese, Edge TTS creates voice, and FFmpeg synchronizes the output
- API keys stay in Streamlit Secrets and are not committed to GitHub

## Streamlit Community Cloud setup

1. Deploy this repository as a Streamlit app with `app.py` as the main file.
2. Keep `packages.txt` containing `ffmpeg`.
3. In **App settings → Secrets**, add:

```toml
APP_PASSWORD = "your-existing-password"
GEMINI_API_KEY = "your-gemini-api-key"
```

Do not put API keys in `app.py` or commit them to GitHub.

## Notes

- URL downloading uses `yt-dlp`; platform availability and copyright restrictions still apply.
- Gemini analyzes the downloaded video for Recap mode, so long videos can use significant quota and memory.
- Whisper `base` multilingual model downloads on first run and uses CPU int8.
- Recap subtitles are exported as an SRT file.
- Streamlit Community Cloud resource limits may constrain long videos.
