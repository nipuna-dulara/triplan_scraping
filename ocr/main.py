import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import google.generativeai as genai
import shutil
import os
import re
import json
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Configure Google Generative AI
genai.configure(api_key="AIzaSyDv2PfZAzlHE9yoU-Pscy2jNSIvDlSLSPw")
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1000,
    "response_mime_type": "text/plain",
}

# Define Google Generative model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="You will be given the data in a bill scanned by an OCR. You have to extract the date, items, amount, and the time in this format: {date: date, amount: amount, items: items, time: time}. Fill with null if values are not readable. Return the response in JSON format."
)

# FastAPI route to process image uploads


@app.post("/extract-info/")
async def extract_info(file: UploadFile = File(...)):
    try:
        # Save the uploaded image file
        file_location = f"./temp_{file.filename}"
        with open(file_location, "wb+") as f:
            shutil.copyfileobj(file.file, f)

        # Use Tesseract OCR to extract text from the image
        extracted_info = pytesseract.image_to_string(Image.open(file_location))

        # Send the extracted text to Google Generative Model
        chat_session = model.start_chat(
            history=[{"role": "model", "parts": [""]}])
        response = chat_session.send_message(extracted_info)

        # Clean and process the extracted response
        input_str = re.sub(r'NaN', 'null', response.text)
        input_str = re.sub(r'[¥]', '', input_str)
        input_str = re.sub(r'(\d+)/(\d+)', r'\1.\2', input_str)
        input_str = re.sub(r'\\\"', '"', input_str)
        input_str = re.sub(r'\\', '', input_str)

        # Extract specific fields using regex
        date = re.search(r'"date":\s*"([^"]+)"', input_str)
        amount = re.search(r'"amount":\s*"([^"]+)"', input_str)
        time = re.search(r'"time":\s*"([^"]+)"', input_str)
        item_names = re.findall(r'"name":\s*"([^"]+)"', input_str)

        # Create the final data structure
        data = {
            'date': date.group(1) if date else None,
            'amount': amount.group(1) if amount else None,
            'items': item_names,
            'time': time.group(1) if time else None
        }

        # Clean up the saved file
        os.remove(file_location)

        # Return the extracted data
        return data

    except Exception as e:
        # Handle any errors that occur during the process
        raise HTTPException(
            status_code=500, detail=f"Error processing the image: {e}")


# To run: uvicorn filename:app --reload
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
