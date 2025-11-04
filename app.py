from fastapi import FastAPI, UploadFile
from fastapi.background import BackgroundTasks
import openai, tempfile, os, uuid, json

openai.api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()
JOBS = {}   # in-memory store; use Supabase or Redis later

def process_file(job_id: str, path: str):
    try:
        with open(path, "rb") as f:
            text = openai.audio.transcriptions.create(model="whisper-1", file=f).text
        summary_prompt = f"Summarize in 5 concise bullet points:\n{text}"
        summary = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": summary_prompt}]
        ).choices[0].message.content
        JOBS[job_id] = {"status": "complete", "transcript": text, "summary": summary}
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}

@app.post("/start_job")
async def start_job(file: UploadFile, background: BackgroundTasks):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(await file.read())
    tmp.close()
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running"}
    background.add_task(process_file, job_id, tmp.name)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    return JOBS.get(job_id, {"status": "not_found"})
