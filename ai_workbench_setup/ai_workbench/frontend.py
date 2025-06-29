import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml
import plotly.io as pio
from utils.logger import setup_logger
from utils.helpers import validate_text
import pandas as pd
import os
import io
import json

logger = setup_logger(__name__)

with open("config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

API_URL = f"http://localhost:{config['api']['port']}/process"
UPLOAD_URL = f"http://localhost:{config['api']['port']}/upload_documents"
VOICE_INPUT_URL = f"http://localhost:{config['api']['port']}/voice_input"
VOICE_OUTPUT_URL = f"http://localhost:{config['api']['port']}/voice_output"
CROWD_URL = f"http://localhost:{config['api']['port']}/crowdsource"
FINE_TUNE_URL = f"http://localhost:{config['api']['port']}/fine_tune"
PENDING_URL = f"http://localhost:{config['api']['port']}/pending_datasets"
APPROVE_URL = f"http://localhost:{config['api']['port']}/approve_dataset"

st.title("AI Workbench")

task = st.selectbox("Select Task", ["summarization", "translation", "chat"])

st.sidebar.header("Model Parameters")
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, config["models"]["llama"]["temperature"])
top_p = st.sidebar.slider("Top-P", 0.1, 1.0, config["models"]["llama"]["top_p"])
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, config["models"]["llama"]["max_tokens"])
min_tokens = st.sidebar.slider("Min Tokens", 10, 100, config["models"]["llama"]["min_tokens"])
use_fusion = st.sidebar.checkbox("Use Model Fusion", value=False)
use_tutor = st.sidebar.checkbox("Use AI Tutor", value=False)

st.sidebar.header("Evaluation Metrics")
available_metrics = {
    "summarization": ["rouge1", "rouge2", "rougeL", "bertscore"],
    "translation": ["bleu", "meteor"],
    "chat": ["perplexity"]
}
metrics = st.sidebar.multiselect("Select Metrics", available_metrics[task], default=available_metrics[task])

st.sidebar.header("Voice Input")
use_voice = st.sidebar.checkbox("Use Voice Input", value=False)
if use_voice:
    voice_file = st.sidebar.file_uploader("Upload Audio (WAV)", type=["wav"])
    if voice_file:
        response = requests.post(VOICE_INPUT_URL, files={"file": voice_file})
        if response.status_code == 200:
            text_input = response.json()["text"]
            st.sidebar.success("Voice input processed")
        else:
            st.sidebar.error("Failed to process voice input")
else:
    text_input = None

if task == "summarization":
    text_input = st.text_area("Enter text to summarize:", value=text_input or "", height=200)
    reference = st.text_area("Enter reference summary (optional):", height=100)
    target_lang = None
    messages = None
elif task == "translation":
    text_input = st.text_area("Enter text to translate:", value=text_input or "", height=200)
    target_lang = st.selectbox("Target Language", ["Spanish", "French", "German"])
    reference = st.text_area("Enter reference translation (optional):", height=100)
    messages = None
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    text_input = st.text_input("Enter your message:", value=text_input or "")
    if st.button("Send"):
        st.session_state.messages.append({"role": "user", "content": text_input})
    reference = None
    target_lang = None
    messages = st.session_state.messages
    for msg in messages:
        st.write(f"**{msg['role'].capitalize()}**: {msg['content']}")

st.sidebar.header("Upload Documents for RAG")
uploaded_file = st.sidebar.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
if uploaded_file and st.sidebar.button("Upload Document"):
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    response = requests.post(UPLOAD_URL, files=files)
    if response.status_code == 200:
        st.sidebar.success("Document uploaded successfully")
    else:
        st.sidebar.error("Failed to upload document")

st.sidebar.header("Contribute Dataset")
dataset_input = st.sidebar.text_area("Enter dataset (JSON format):")
submitter = st.sidebar.text_input("Your Name (optional):", value="anonymous")
if st.sidebar.button("Submit Dataset"):
    try:
        data = json.loads(dataset_input)
        response = requests.post(CROWD_URL, json={"data": data, "submitter": submitter})
        if response.status_code == 200:
            st.sidebar.success("Dataset submitted for review")
        else:
            st.sidebar.error("Failed to submit dataset")
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON format")

st.sidebar.header("Moderate Datasets")
if st.sidebar.button("View Pending Datasets"):
    response = requests.get(PENDING_URL)
    if response.status_code == 200:
        datasets = response.json()["datasets"]
        for dataset in datasets:
            st.sidebar.write(f"ID: {dataset['id']}, Submitter: {dataset['submitter']}")
            if st.sidebar.button(f"Approve Dataset {dataset['id']}"):
                response = requests.post(f"{APPROVE_URL}/{dataset['id']}")
                if response.status_code == 200:
                    st.sidebar.success(f"Dataset {dataset['id']} approved")
                else:
                    st.sidebar.error("Failed to approve dataset")

st.sidebar.header("Fine-Tune Model")
fine_tune_file = st.sidebar.file_uploader("Upload fine-tuning dataset (JSON)", type=["json"])
if fine_tune_file and st.sidebar.button("Fine-Tune"):
    files = {"file": (fine_tune_file.name, fine_tune_file, "application/json")}
    response = requests.post(FINE_TUNE_URL, files=files)
    if response.status_code == 200:
        st.sidebar.success("Model fine-tuned successfully")
    else:
        st.sidebar.error("Failed to fine-tune model")

if (task in ["summarization", "translation"] and st.button("Process")) or (task == "chat" and messages):
    try:
        payload = {
            "task": task,
            "text": text_input if text_input else "",
            "reference": reference if reference else "",
            "target_lang": target_lang if target_lang else "",
            "messages": messages if messages else [],
            "params": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(max_tokens),
                "min_tokens": int(min_tokens)
            },
            "metrics": metrics if metrics else [],
            "use_fusion": use_fusion,
            "use_tutor": use_tutor
        }
        logger.info(f"Sending payload to {API_URL}: {json.dumps(payload, indent=2)}")
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=[502, 503, 504, 10054])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        response = session.post(API_URL, json=payload, timeout=600)
        logger.info(f"Response status: {response.status_code}")
        response.raise_for_status()
        if task == "chat":
            data = response.text  # Plain text response
            logger.info(f"Received response: {data}")
        else:
            data = response.json()
            logger.info(f"Received response: {json.dumps(data, indent=2)}")

        st.subheader("Results")
        if task == "chat":
            st.write(f"**GPT-4o**: {data if data else 'No output'}")
        else:
            for result in data.get("results", []):
                st.write(f"**{result['model']}**: {result.get('output', result.get('error', 'No output'))}")

        if task != "chat" and data.get("evaluation"):
            st.subheader("Evaluation Metrics")
            eval_df = pd.DataFrame(data["evaluation"])
            st.dataframe(eval_df)

    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        st.error(f"ValueError: {str(ve)}")
    except requests.RequestException as re:
        logger.error(f"API request failed: {str(re)}")
        st.error(f"API request failed: {str(re)}")
    except Exception as e:
        logger.error(f"Frontend error: {str(e)}")
        st.error(f"Frontend error: {str(e)}")

st.subheader("Collaboration Arena")
arena_task = st.selectbox("Select Arena Task", ["summarization", "translation", "chat"], key="arena_task")
arena_input = st.text_area("Enter Arena Input:", height=100, key="arena_input")
if st.button("Join Arena"):
    st.write("Connected to arena. Waiting for results...")