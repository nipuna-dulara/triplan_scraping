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
    suggestions = data.Title.iloc[top_n_indices]

    return suggestions


description = "i want to go see wildlife and nature. and to enjoy safari"
suggested_places = suggest_places(description, num_suggestions=4)
print(suggested_places)
