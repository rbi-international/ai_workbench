import pytest
from src.tasks.summarizer import Summarizer
from src.tasks.translator import Translator
from src.tasks.chatter import Chatter
from src.tutor import AITutor
from src.crowdsource import CrowdsourceManager
from src.voice import VoiceProcessor
from src.fusion import ModelFusion
from src.ethics import EthicsAnalyzer
from src.models.llama_model import LlamaModel
from src.models.openai_model import OpenAIModel
import yaml
import pandas as pd

with open("config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

models = [
    LlamaModel(config["models"]["llama"]),
    OpenAIModel(config["models"]["openai"])
]
params = config["models"]["llama"]

def test_summarizer():
    summarizer = Summarizer(models)
    results = summarizer.summarize("This is a test text.", params)
    assert len(results) == len(models)
    assert all("model" in r and "output" in r and "inference_time" in r for r in results)

def test_translator():
    translator = Translator(models)
    results = translator.translate("Hello world", "Spanish", params)
    assert len(results) == len(models)
    assert all("model" in r and "output" in r and "inference_time" in r for r in results)

def test_chatter():
    chatter = Chatter(models)
    messages = [{"role": "user", "content": "Hi"}]
    results = chatter.chat(messages, params)
    assert len(results) == len(models)
    assert all("model" in r and "output" in r and "inference_time" in r for r in results)

def test_tutor():
    tutor = AITutor()
    results = [{"model": "test", "output": "Test summary", "inference_time": 1.0}]
    evaluation = pd.DataFrame([{"model": "test", "rouge1": 0.8, "inference_time": 1.0}])
    explanation = tutor.explain_performance("summarization", results, evaluation)
    assert isinstance(explanation, str)
    assert "rouge1" in explanation.lower()

def test_crowdsource():
    crowdsource = CrowdsourceManager()
    data = [{"text": "test", "label": "test"}]
    dataset_id = crowdsource.submit_dataset(data, "test_user")
    assert isinstance(dataset_id, int)
    pending = crowdsource.get_pending_datasets()
    assert any(d["submitter"] == "test_user" for d in pending)

def test_voice():
    voice = VoiceProcessor()
    assert isinstance(voice, VoiceProcessor)

def test_fusion():
    fusion = ModelFusion()
    results = [
        {"model": "model1", "output": "Summary one", "inference_time": 1.0},
        {"model": "model2", "output": "Summary two", "inference_time": 1.0}
    ]
    fused = fusion.fuse_outputs(results, "Reference summary")
    assert "fused_output" in fused
    assert "weights" in fused

def test_ethics():
    ethics = EthicsAnalyzer()
    results = ethics.analyze(["This is a test", "This is offensive"])
    assert len(results) == 2
    assert all("sentiment" in r and "toxicity" in r for r in results)