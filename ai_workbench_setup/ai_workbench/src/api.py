from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket
from pydantic import BaseModel
from src.tasks.summarizer import Summarizer
from src.tasks.translator import Translator
from src.tasks.chatter import Chatter
from src.rag.retriever import RAGRetriever
from src.evaluator import Evaluator
from src.fine_tuner import FineTuner
from src.tutor import AITutor
from src.crowdsource import CrowdsourceManager
from src.voice import VoiceProcessor
from src.fusion import ModelFusion
from src.collaboration import CollaborationArena
from src.ethics import EthicsAnalyzer
from utils.logger import setup_logger
from utils.visualizer import Visualizer
from utils.explainability import Explainability
from src.models.llama_model import LlamaModel
from src.models.openai_model import OpenAIModel
import yaml
import PyPDF2
import pytesseract
from PIL import Image
import io
from typing import List, Dict, Any, Optional
import json

app = FastAPI(title="AI Workbench API")
logger = setup_logger(__name__)

with open("config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

models = [
    # LlamaModel(config["models"]["llama"]),
    OpenAIModel(config["models"]["openai"])
]
summarizer = Summarizer(models)
translator = Translator(models)
chatter = Chatter(models)
evaluator = Evaluator()
retriever = RAGRetriever()
visualizer = Visualizer()
# fine_tuner = FineTuner()
tutor = AITutor()
crowdsource = CrowdsourceManager()
voice_processor = VoiceProcessor()
fusion = ModelFusion()
collaboration = CollaborationArena()
ethics = EthicsAnalyzer()
explainability = Explainability()

class TaskInput(BaseModel):
    task: str
    text: Optional[str] = None
    reference: Optional[str] = None
    target_lang: Optional[str] = None
    messages: Optional[List[Dict]] = None
    params: Dict[str, Any] = {}
    metrics: List[str] = []
    use_fusion: bool = False
    use_tutor: bool = False

class CrowdsourceInput(BaseModel):
    data: List[Dict]
    submitter: str = "anonymous"

@app.post("/process")
async def process_task(input: TaskInput):
    try:
        logger.info(f"Received request payload: {input.dict()}")
        params = {
            "max_tokens": input.params.get("max_tokens", config["models"]["llama"]["max_tokens"]),
            "min_tokens": input.params.get("min_tokens", config["models"]["llama"]["min_tokens"]),
            "temperature": input.params.get("temperature", config["models"]["llama"]["temperature"]),
            "top_p": input.params.get("top_p", config["models"]["llama"]["top_p"])
        }

        logger.info("Step 1: Retrieving context")
        context = []
        if input.text:
            context = retriever.retrieve(input.text)
        logger.info(f"Step 2: Context retrieved: {context}")

        if input.task == "summarization":
            logger.info("Step 3: Processing summarization")
            text = input.text + "\nContext: " + "\n".join([doc["text"] for doc in context])
            results = summarizer.summarize(text, params)
            logger.info(f"Step 4: Summarization results: {results}")
            evaluation = evaluator.evaluate_summarization(results, input.reference) if input.reference else None
            return {
                "results": results,
                "evaluation": evaluation.to_dict(orient="records") if evaluation is not None else None
            }
        elif input.task == "translation":
            logger.info("Step 3: Processing translation")
            text = input.text + "\nContext: " + "\n".join([doc["text"] for doc in context])
            results = translator.translate(text, input.target_lang, params)
            logger.info(f"Step 4: Translation results: {results}")
            evaluation = evaluator.evaluate_translation(results, input.reference) if input.reference else None
            return {
                "results": results,
                "evaluation": evaluation.to_dict(orient="records") if evaluation is not None else None
            }
        elif input.task == "chat":
            logger.info("Step 3: Processing chat")
            messages = input.messages or [{"role": "user", "content": input.text}]
            if context:
                messages[0]["content"] += "\nContext: " + "\n".join([doc["text"] for doc in context])
            results = chatter.chat(messages, params)
            logger.info(f"Step 4: Chat results: {results}")
            return results[0]["output"]  # Plain text response
        else:
            raise ValueError(f"Unsupported task: {input.task}")
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/upload_documents")
async def upload_documents(file: UploadFile = File(...)):
    try:
        if file.filename.endswith(".pdf"):
            pdf = PyPDF2.PdfReader(file.file)
            text = "".join([page.extract_text() for page in pdf.pages])
        elif file.filename.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(file.file)
            text = pytesseract.image_to_string(image)
        else:
            raise ValueError("Unsupported file type")
        retriever.add_documents([text])
        return {"message": "Document uploaded and indexed"}
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice_input")
async def voice_input(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()
        text = voice_processor.speech_to_text(audio_data)
        return {"text": text}
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice_output")
async def voice_output(text: str):
    try:
        output_path = voice_processor.text_to_speech(text)
        return {"audio_path": output_path}
    except Exception as e:
        logger.error(f"Error generating voice output: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crowdsource")
async def crowdsource_dataset(input: CrowdsourceInput):
    try:
        dataset_id = crowdsource.submit_dataset(input.data, input.submitter)
        return {"dataset_id": dataset_id, "message": "Dataset submitted for review"}
    except Exception as e:
        logger.error(f"Error submitting crowdsourced dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pending_datasets")
async def get_pending_datasets():
    try:
        datasets = crowdsource.get_pending_datasets()
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Error fetching pending datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/approve_dataset/{dataset_id}")
async def approve_dataset(dataset_id: int):
    try:
        crowdsource.approve_dataset(dataset_id)
        return {"message": f"Dataset {dataset_id} approved"}
    except Exception as e:
        logger.error(f"Error approving dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            try:
                await collaboration.broadcast_results(data.get("task", "unknown"), data.get("results", []))
                await websocket.send_json({"status": "received"})
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()