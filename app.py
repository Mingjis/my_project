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
    save_path = "fine_tuned_model.pt"
    github_download_url = "https://github.com/Mingjis/my_project/releases/download/v1.0/fine_tuned_model.pt"

    if not os.path.exists(save_path):
        response = requests.get(github_download_url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

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

def load_law_risks():
    law_risk_dict = {}
    risks_path = "law_risks.txt"

    if os.path.exists(risks_path):
        with open(risks_path, "r", encoding="utf-8") as f:
            for line in f:
                if " , " in line and " / " in line:
                    key, risks = line.split(" , ", 1)
                    physical_risks, human_risks = risks.strip().split(" / ")
                    law_risk_dict[key.strip()] = {
                        "physical": set(physical_risks.split(";")),
                        "human": set(human_risks.split(";")),
                    }
    return law_risk_dict

def filter_laws_by_risks(physical_risk, human_risk, law_risks):
    filtered_laws = []
    for law, risks in law_risks.items():
        if (physical_risk in risks["physical"]) or (human_risk in risks["human"]):
            filtered_laws.append(law)
    return filtered_laws

model = load_model()
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
label_encoder = joblib.load("label_encoder.pkl")
law_details = load_law_details()
law_risks = load_law_risks()

st.title("Safety Legislation Recommendation for DfS report")

physical_risk = st.text_input("물적위험성(필수) :")
human_risk = st.text_input("인적위험성(필수) :")
keyword1 = st.text_input("Keyword 1(필수) :")
keyword2 = st.text_input("Keyword 2(선택) :")
keyword3 = st.text_input("Keyword 3(선택) :")

if st.button("Search"):
    keywords = [keyword1, keyword2, keyword3]
    input_text = f"{physical_risk} {human_risk} " + " ".join([k for k in keywords if k])

    filtered_laws = filter_laws_by_risks(physical_risk, human_risk, law_risks)

    if not filtered_laws:
        st.warning("")
    else:
        st.write(f"")

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
            if law in filtered_laws:
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
