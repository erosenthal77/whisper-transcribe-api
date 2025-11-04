from fastapi import FastAPI, Form
import requests, time, openai, os

ASSEMBLY_KEY = os.getenv("ASSEMBLYAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

@app.post("/transcribe")
async def transcribe(url: str = Form(...)):
    """Transcribe long YouTube or audio URLs asynchronously."""
    try:
        # 1️⃣ Submit job to AssemblyAI
        headers = {"authorization": ASSEMBLY_KEY, "content-type": "application/json"}
        payload = {"audio_url": url}
        resp = requests.post("https://api.assemblyai.com/v2/transcript", json=payload, headers=headers)
        job_id = resp.json()["id"]

        # 2️⃣ Poll until done
        while True:
            status = requests.get(f"https://api.assemblyai.com/v2/transcript/{job_id}", headers=headers).json()
            if status["status"] == "completed":
                transcript = status["text"]
                break
            if status["status"] == "error":
                return {"error": status["error"]}
            time.sleep(5)  # wait 5 seconds before checking again

        # 3️⃣ Summarize with GPT-5
        summary_prompt = f"Summarize this transcript in 5 concise bullet points:\n{transcript}"
        summary = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": summary_prompt}]
        ).choices[0].message.content

        return {"transcript": transcript, "summary": summary}

    except Exception as e:
        return {"error": str(e)}
