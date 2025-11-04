from fastapi import FastAPI, Body
import openai, tempfile, subprocess, os, shlex

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/transcribe")
async def transcribe(data: dict = Body(...)):
    """Accept JSON: { "url": "<youtube link>" }"""
    url = data.get("url")
    if not url:
        return {"error": "Provide a YouTube URL in JSON body, e.g. {'url': 'https://...'}"}

    try:
        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        cmd = (
            f"yt-dlp -x --audio-format mp3 "
            f"--no-playlist --quiet --no-warnings "
            f"--max-filesize 50M -o {shlex.quote(tmp_audio.name)} {shlex.quote(url)}"
        )
        subprocess.run(cmd, shell=True, timeout=60, check=True)

        with open(tmp_audio.name, "rb") as f:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f
            ).text

        summary = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": f"Summarize this transcript in 5 concise bullets:\n{transcript}"}]
        ).choices[0].message.content

        return {"transcript": transcript, "summary": summary}

    except subprocess.TimeoutExpired:
        return {"error": "Download timed out — try a shorter clip (<60 s)."}
    except subprocess.CalledProcessError:
        return {"error": "YouTube download failed — check the link."}
    except Exception as e:
        return {"error": str(e)}
