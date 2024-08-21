from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('places_data_new2.csv')
# Add a numeric label column
print(data.head())
# data['Label'] = pd.factorize(data['Title'])[0]
# data.to_csv('places_data_new2.csv', index=False)
# # Quick look at the data
# print(data.head())

# # Distribution of ratings
# sns.histplot(data['Rating'], kde=True)
# plt.title('Distribution of Ratings')
# plt.show()

# # Distribution of ideal durations
# sns.histplot(data['Ideal Duration'], kde=True)
# plt.title('Distribution of Ideal Durations')
# plt.show()

# # Best Time to visit
# print(data['Best Time'].value_counts())

# # Example of text data
# print(data['Short Description'].iloc[0])
# print(data['Paragraphs'].iloc[0])


# # Convert descriptions to TF-IDF vectors
# vectorizer = TfidfVectorizer(max_features=5000)
# tfidf_matrix = vectorizer.fit_transform(data['Short Description'])

# # Compute similarity between all places
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# # Get the top 5 most similar places for the first place in the dataset
# similar_indices = cosine_sim[0].argsort()[-6:-1]
# similar_places = data.iloc[similar_indices]

# print("Top similar places:")
# print(similar_places[['Title', 'Short Description']])
