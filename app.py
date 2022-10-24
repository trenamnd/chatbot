import time

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoModel, AutoTokenizer
from loguru import logger

app = Flask(__name__)
CORS(app)

model, tokenizer = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"), AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2")

questions_to_answers = {
    "What is the name of the football team": "Bearcats",
    "What is the name of the engineering school":
    "CEAS (College of Engineering and Applied Science)",
    "What is the name of the business school":
    "Carl H. Lindner College of Business",
    "Who is the president of the university": "Dr. Neville Pinto",
    "Who is the football coach": "Luke Fickell",
    "What are the on-campus housing options":
    "Apartment-style, traditional dorms, and Greek housing",
    "Where is the gym": "The CRC (Campus Recreation Center)",
    "What is the name of the mascot": "Bearcat",
    "How many years is an CS degree": "5 years",
    "How many co-op terms":
    "The UC engineering program has 3 to 5 co-op terms",
    "Is there a computer science major": "Yes, of course!"
}


def embed_text(text: str):
  return model(
      **tokenizer([text], return_tensors="pt", padding=True,
                  truncation=True)).pooler_output[0].detach().numpy()


questions_embeddings = np.array(
    [embed_text(question) for question in questions_to_answers])


def best_match_index(embed, embeddings):
  return np.argmax(np.dot(embed, embeddings.T))


@app.route('/query', methods=['POST'])
def query():
  query = request.json['query']
  embedding = embed_text(query)
  best_match = best_match_index(embedding, questions_embeddings)
  question = list(questions_to_answers.keys())[best_match]
  answer = list(questions_to_answers.values())[best_match]
  return jsonify({'question': question, 'answer': answer})


@app.route("/")
def index():
  sent = send_from_directory("./", "index.html")
  sent.last_modified = time.time()
  return sent


if __name__ == '__main__':
  app.run(port=8000, debug=True)
