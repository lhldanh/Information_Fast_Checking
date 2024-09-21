from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.TextPreprocess import sent_tokenize_and_preprocess
from utils.TextPreprocess import sentence_preprocess
from underthesea import sent_tokenize
import numpy as np
import torch

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = "MoritzLaurer/ernie-m-large-mnli-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    model_name).to(device)

model_name = "MoritzLaurer/ernie-m-large-mnli-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    model_name).to(device)


def NLI(premise, hypothesis):
    inp = tokenizer(premise, hypothesis,
                    truncation=True, return_tensors="pt")
    output = nli_model(inp["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["Thông tin chính xác", "Không tìm ra thông tin trong đoạn văn bản", "Thông tin sai lệch"]
    prediction = {name: round(float(pred) * 100, 1)
                  for pred, name in zip(prediction, label_names)}
    return prediction


def get_max_similarity_score_sentence(sentence, paragraph, model):

    # Embedding
    paragraph_old = sent_tokenize(paragraph)
    sentence = sentence_preprocess(sentence)
    paragraph = sent_tokenize_and_preprocess(paragraph)
    sentence_embedding = model.encode(sentence)
    paragraph_embedding = model.encode(paragraph)

    # Compute similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [cosine_similarity(
        sent, sentence_embedding) for sent in paragraph_embedding]

    most_similar_index = np.argmax(similarities)

    return paragraph_old[most_similar_index]
