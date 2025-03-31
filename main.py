from fastapi import FastAPI, Form, UploadFile, File
import json
import os

from functions import *  # Make sure all ga1_qX functions are defined here
from sent_tranf import MatchQuestion  # If you're using sentence transformers for question matching

app = FastAPI()

# Load question-to-function mapping
with open("question_mapper.json", "r") as f:
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
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(contents)

    try:
        result = handler(question, file_path)
        return {"answer": result}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}
