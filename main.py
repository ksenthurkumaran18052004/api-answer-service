from fastapi import FastAPI, UploadFile, File, Form

app = FastAPI()

@app.post("/api/")
async def solve_question(question: str = Form(...), file: UploadFile = File(None)):
    return {"answer": "Test successful!"}
