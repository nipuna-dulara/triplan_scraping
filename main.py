import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialize the FastAPI app
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset
data = pd.read_csv('places_data_new2.csv')

# Combine text fields to form the input for the model
data['input_text'] = (
    data['Title'].fillna('') + ' ' +
    data['Short Description'].fillna('') + ' ' +
    data['Paragraphs'].fillna('') + ' ' +
    data['Hotels'].fillna('') + ' ' +
    data['Things to do'].fillna('')
)

# Ensure all entries in 'input_text' are strings
data['input_text'] = data['input_text'].fillna('').astype(str)


def get_image_path(title):
    base_path = f"static/images/{title}"
    image_path = os.path.join(base_path, "0.jpg")
    if not os.path.exists(image_path):
        image_path = os.path.join("static/images", "default.jpg")
    return "http://127.0.0.1:8000/"+image_path


data['image_path'] = data['Title'].apply(get_image_path)
# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['input_text'])


def suggest_places(description, num_suggestions=3):
    # Vectorize the user's input description
    description_vec = tfidf_vectorizer.transform([description])

    # Calculate cosine similarity between the user's input and all place descriptions
    similarity_scores = cosine_similarity(
        description_vec, tfidf_matrix).flatten()

    # Get the indices of the top N similar places
    top_n_indices = similarity_scores.argsort()[-num_suggestions:][::-1]

    # Get the titles of the top N similar places
    suggestions = data[['Title', 'image_path']].iloc[top_n_indices]

    return suggestions


# description = "i want to go see wildlife and nature. and to enjoy safari"
# suggested_places = suggest_places(description, num_suggestions=4)
# print(suggested_places)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Allow all origins for testing; in production, restrict this.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Configure the Generative AI model
genai.configure(api_key="AIzaSyDv2PfZAzlHE9yoU-Pscy2jNSIvDlSLSPw")

# Define the chat model settings
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="you are chat bot in a trip planning app and you are chatting with a user who is planning a trip. app name is triplan. you will receive a description and suggested places in this format {description , places : place1, place2, place3} you have to elaborate on the places. one by one."
)
chat_session = model.start_chat(
    history=[
        {
            "role": "model",
            "parts": [
                "Hello! welcome to triplan. How can I help you plan your trip today?",
            ],
        },
    ]
)


class UserInput(BaseModel):
    message: str


class DescriptionRequest(BaseModel):
    description: str
    num_suggestions: int = 3


@app.post("/chat/")
async def chat_with_bot(user_input: UserInput):
    print(user_input.message)
    try:
        suggested_places = suggest_places(
            user_input.message, num_suggestions=4)
        print(suggested_places)
        places_str = ", ".join(suggested_places['Title'])
        # Format the message for the chatbot

        message = f"{user_input.message}, places: {places_str}"

        # Send the user's message to the chat model
        response = chat_session.send_message(message)

        # Include the titles and image paths in the response
        response_data = {
            "response": response.text,
            "suggested_places": suggested_places.to_dict(orient="records")
        }

        return response_data
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error interacting with chatbot: {str(e)}")

# Run the app (this is useful if you're running the script directly)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
