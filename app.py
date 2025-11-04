from fastapi import FastAPI, UploadFile, Form
import openai, tempfile, subprocess, os

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/transcribe")
async def transcribe(file: UploadFile = None, url: str = Form(None)):
    try:
        # 1. Download audio if URL
        if url and not file:
            tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            subprocess.run([
                "yt-dlp", "-x", "--audio-format", "mp3",
                "-o", tmp_audio.name, url
            ])
            audio_path = tmp_audio.name
        else:
            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            tmp_file.write(await file.read())
            tmp_file.close()
            audio_path = tmp_file.name

        # 2. Whisper transcription
        with open(audio_path, "rb") as f:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f
            ).text

        # 3. GPT summary
        summary_prompt = f"Summarize this transcript in 5 concise bullet points:\n{transcript}"
        summary = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": summary_prompt}]
        ).choices[0].message.content

        return {"transcript": transcript, "summary": summary}

    except Exception as e:
        return {"error": str(e)}
