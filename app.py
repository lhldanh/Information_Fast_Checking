import streamlit as st
from sentence_transformers import SentenceTransformer
from utils.FastChecking import get_max_similarity_score_sentence, NLI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

STransModel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model_name = "MoritzLaurer/ernie-m-large-mnli-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    model_name).to(device)


def process_result(pred):
    for key in pred:
        if pred[key] >= 50:
            return key, pred[key]


if __name__ == "__main__":
    st.title(':blue[VNese Information Fast Checking]')

    sentence = st.text_input(
        'Nhập thông tin cần kiểm chứng',
        placeholder='Chỉ 1 câu')
    paragraph = st.text_input('Nhập đoạn văn cần kiểm chứng')

    if st.button('Kiểm tra'):
        most_similar_sentence = get_max_similarity_score_sentence(
            sentence, paragraph, model=STransModel)

        col1, col2 = st.columns(2)
        with col1:
            st.write(':blue[Kết quả kiểm tra]')
            prediction = NLI(sentence, most_similar_sentence)
            res = process_result(prediction)
            st.write(f':red[{res[0]}]')
            st.write(f'Độ chính xác của kết quả: {res[1]}')
        with col2:
            st.write(':blue[Dẫn chứng]')
            st.write(most_similar_sentence)
