import unicodedata
import string
from underthesea import sent_tokenize
from pyvi.ViTokenizer import tokenize


def read_text_from_file(file_path):

    with open(file_path, encoding='utf8') as f:
        text = f.read()
        return text


stopwords = read_text_from_file('utils\stopwords.txt').split('\n')
# stopwords = []


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)

    return text.translate(translator)


def sentence_preprocess(sentence):
    sentence = sentence.lower()
    sentence = remove_punctuation(sentence)
    sentence = unicodedata.normalize('NFC', sentence)
    sentence = tokenize(sentence)
    # sentence = [word for word in sentence if word not in stopwords]
    sentence = ' '.join(['_'.join(word.split(' ')) for word in sentence])

    return sentence


def sent_tokenize_and_preprocess(paragraph):
    paragraph = sent_tokenize(paragraph)
    paragraph = [sentence_preprocess(sentence) for sentence in paragraph]

    return paragraph
