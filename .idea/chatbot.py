import json
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

with open('intent.json') as file:
    intents=json.load(file)

stemmer=PorterStemmer()
stop_words=set(stopwords.words('english'))

def preprocess_text(text):
    text=text.lower()
    text=text.translate(str.maketrans('', '', string.punctuation))
    tokens=text.split()
    tokens=[stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_data(intents):
    patterns = []
    tags = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            patterns.append(preprocess_text(pattern))
            tags.append(intent['tag'])
    return patterns, tags

patterns, tags=preprocess_data(intents)
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(patterns)

def get_best_response(user_input, threshold=0.2):
    user_input_processed=preprocess_text(user_input)
    user_vector=vectorizer.transform([user_input_processed])
    similarities=cosine_similarity(user_vector, X)
    max_similarity=np.max(similarities)

    if max_similarity>threshold:
        best_match_index=np.argmax(similarities)
        best_tag=tags[best_match_index]
        return best_tag
    return None

def split_questions(user_input):
    separators=["and", ".", "?", "!"]
    user_input = user_input.lower()
    for sep in separators:
        user_input=user_input.replace(sep, "|")
    questions=user_input.split("|")
    return [q.strip() for q in questions if q.strip()]

def get_response(user_input):
    questions = split_questions(user_input)
    responses = []
    for question in questions:
        best_tag=get_best_response(question)
        if best_tag:
            for intent in intents['intents']:
                if intent['tag'] == best_tag:
                    responses.append(random.choice(intent['responses']))
        else:
            responses.append("Sorry, I didn't understand that.")
    return " ".join(responses)

def main():
    print("Hello! You can start chatting with the bot. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower()=='exit':
            break
        response=get_response(user_input)
        print("Bot: " + response)

if __name__ == "__main__":
    main()
