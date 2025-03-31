from fastapi import FastAPI, Form, UploadFile, File
from functions import *
from sent_tranf import MatchQuestion
import json

app = FastAPI()

with open("question_mapper.json", 'r') as f:
    question_mapper = json.load(f)

matcher = MatchQuestion()

@app.post("/api/")
async def solve_question(question: str = Form(...), file: UploadFile = File(None)):
    matched_q = matcher.match_question(question)
    handler_name = question_mapper.get(matched_q)

    if not handler_name or handler_name not in globals():
        return {"answer": "No handler found for this question."}

    handler = globals()[handler_name]

    file_path = None
    if file:
        contents = await file.read()
        file_path = f"uploads/{file.filename}"
        with open(file_path, 'wb') as f:
            f.write(contents)

    try:
        answer = handler(question, file_path)
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}
# Required by Vercel
handler = app
