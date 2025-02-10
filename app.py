import streamlit as st
import torch
import joblib
from transformers import ElectraTokenizer, ElectraModel
import os
import requests
import re

class KoELECTRAClassifier(torch.nn.Module):
    def __init__(self, electra, output_size):
        super(KoELECTRAClassifier, self).__init__()
        self.electra = electra
        self.fc = torch.nn.Linear(electra.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(pooled_output)

@st.cache_resource
def load_model():
    """GitHub Releases에서 모델 다운로드 후 로드"""
    save_path = "fine_tuned_model.pt"
    github_download_url = "https://github.com/Mingjis/my_project/releases/download/v1.0/fine_tuned_model.pt"

    if not os.path.exists(save_path):
        st.write("📥 모델을 GitHub Releases에서 다운로드 중...")
        response = requests.get(github_download_url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.write("✅ 모델 다운로드 완료!")

    try:
        electra_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        model = KoELECTRAClassifier(electra=electra_model, output_size=412)

        model = torch.load(save_path, map_location="cpu")
        model.eval()
        return model
    except RuntimeError as e:
        st.error(f"❌ 모델 로드 중 오류 발생: {e}")
        raise

def load_law_details():
    """ law_details.txt 파일을 로드하여 법령명과 법령 내용을 매핑 """
    law_dict = {}
    details_path = "law_details.txt"

    if os.path.exists(details_path):
        with open(details_path, "r", encoding="utf-8") as f:
            for line in f:
                if " , " in line:
                    key, value = line.split(" , ", 1)
                    key = key.strip() + " "
                    law_dict[key] = value.strip()
    return law_dict

model = load_model()
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
label_encoder = joblib.load("label_encoder.pkl")
law_details = load_law_details()

st.title("Safety Legislation Recommendation for DfS report")

physical_risk = st.text_input("물적 위험성(필수) :")
human_risk = st.text_input("인적 위험성(필수) :")
keyword1 = st.text_input("Keyword 1(필수) :")
keyword2 = st.text_input("Keyword 2(선택) :")
keyword3 = st.text_input("Keyword 3(선택) :")

if st.button("Search"):
    keywords = [keyword1, keyword2, keyword3]
    input_text = f"{physical_risk} {human_risk} " + " ".join([k for k in keywords if k])

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=64)

    with torch.no_grad():
        logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_k_probs, top_k_indices = torch.topk(probs, k=3, dim=1)

    top_k_indices = top_k_indices.squeeze().tolist()
    if not isinstance(top_k_indices, list):
        top_k_indices = [top_k_indices]

    predicted_laws = label_encoder.inverse_transform(top_k_indices)

    unique_laws = set()
    final_laws = []

    for law in predicted_laws:
        split_laws = re.split(r"(?=제\d+조)", law)
        for l in split_laws:
            cleaned_law = l.strip()
            if cleaned_law and cleaned_law not in unique_laws:
                unique_laws.add(cleaned_law)
                final_laws.append(cleaned_law)

    st.subheader("List of Recommendation:")
    for i, law in enumerate(final_laws, start=1):
        cleaned_law = law.strip() + " "

        if cleaned_law in law_details:
            law_detail = law_details[cleaned_law]
        else:
            st.write(f"🔍 '{cleaned_law}'을(를) 찾을 수 없음 → law_details.keys()와 비교 필요")
            st.write(f"🔍 현재 저장된 키 값 샘플: {list(law_details.keys())[:10]}")
            law_detail = "관련 내용 없음"

        st.write(f"{i}. {cleaned_law} - {law_detail}")
