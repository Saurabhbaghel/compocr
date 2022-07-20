from fastapi import FastAPI, Query, Path, File, UploadFile
from pipeline import main

description = """
compocr is the combination of DocTR and Pytesseract. DocTR detects the text in the image and Pytesseract identifies it and converts it into string.
"""

app = FastAPI(
    title = 'compocr api'
)

@app.post('/upload')
async def upload_img(File: UploadFile):
    out = main(File)
    return out