from fastapi import FastAPI, UploadFile
import openai, tempfile, os

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    """Works on Render Starter: processes the uploaded video right away."""
    try:
        # Save upload to a temp file
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(await file.read())
        tmp.close()

        # Whisper transcription
        with open(tmp.name, "rb") as f:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f
            ).text

        # Summarize the transcript
        summary_prompt = f"Summarize this transcript in 5 concise bullet points:\n{transcript}"
        summary = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": summary_prompt}]
        ).choices[0].message.content

        return {"transcript": transcript, "summary": summary}

    except Exception as e:
        return {"error": str(e)}
