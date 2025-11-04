from fastapi import FastAPI, UploadFile, Form, Body
import openai, tempfile, subprocess, os

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/transcribe")
async def transcribe(file: UploadFile = None, url: str = Body(None)):
    try:
        # 1️⃣ Determine input source
        if url and not file:
            tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            subprocess.run([
                "yt-dlp", "-x", "--audio-format", "mp3",
                "-o", tmp_audio.name, url
            ], check=True)
            audio_path = tmp_audio.name
        elif file:
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(await file.read())
            tmp.close()
            audio_path = tmp.name
        else:
            return {"error": "Please provide either a file or a YouTube URL."}

        # 2️⃣ Transcribe
        with open(audio_path, "rb") as f:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f
            ).text

        # 3️⃣ Summarize
        summary_prompt = f"Summarize this transcript in 5 concise bullet points:\n{transcript}"
        summary = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": summary_prompt}]
        ).choices[0].message.content

        return {"transcript": transcript, "summary": summary}

    except subprocess.CalledProcessError as e:
        return {"error": f"YouTube download failed: {e}"}
    except Exception as e:
        return {"error": str(e)}
