from fastapi import FastAPI, UploadFile, Body
import openai, tempfile, subprocess, os, shlex

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/transcribe")
async def transcribe(file: UploadFile = None, url: str = Body(None)):
    try:
        if url and not file:
            tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            cmd = (
                f"yt-dlp -x --audio-format mp3 "
                f"--no-playlist --quiet --no-warnings "
                f"--max-filesize 50M -o {shlex.quote(tmp_audio.name)} {shlex.quote(url)}"
            )
            subprocess.run(cmd, shell=True, timeout=60, check=True)
            audio_path = tmp_audio.name
        elif file:
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(await file.read())
            tmp.close()
            audio_path = tmp.name
        else:
            return {"error": "Provide a file or a YouTube URL."}

        with open(audio_path, "rb") as f:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", file=f
            ).text

        summary = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": f"Summarize in 5 bullets:\n{transcript}"}]
        ).choices[0].message.content

        return {"transcript": transcript, "summary": summary}
    except subprocess.TimeoutExpired:
        return {"error": "Download timed out — try a shorter or smaller video."}
    except subprocess.CalledProcessError as e:
        return {"error": f"YouTube download failed: {e}"}
    except Exception as e:
        return {"error": str(e)}
