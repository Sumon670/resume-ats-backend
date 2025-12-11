from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz
import google.generativeai as genai
import os
from dotenv import load_dotenv
import ssl

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def extract_text_from_pdf(file: UploadFile):
    file_bytes = await file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.post("/calculate-ats-score")
async def calculate_ats_score(
    jd: str = Form(...),
    resume: UploadFile = File(...)
):
    try:
        input_prompt3 = """
        You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, your task is to evaluate the resume against the provided job description. give me the percentage of match i.e. ats score if the resume matches the job description. First the output should come as percentage i.e. ats score and then keywords missing and last final thoughts.
        """
        resume_text = await extract_text_from_pdf(resume)
        response = genai.GenerativeModel('gemini-2.5-flash').generate_content([
            {"text": input_prompt3},
            {"text": f"Job Description:\n{jd}"},
            {"text": f"Resume:\n{resume_text}"}
        ])
        return {"ats_score_response": response.text}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
