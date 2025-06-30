from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import os
import yaml
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import components with error handling
try:
    from src.tasks.summarizer import Summarizer
    from src.tasks.translator import Translator
    from src.tasks.chatter import Chatter
except ImportError as e:
    print(f"Warning: Could not import task modules: {e}")
    Summarizer = Translator = Chatter = None

try:
    from src.rag.retriever import RAGRetriever
except ImportError as e:
    print(f"Warning: Could not import RAG module: {e}")
    RAGRetriever = None

try:
    from src.evaluator import Evaluator
except ImportError as e:
    print(f"Warning: Could not import Evaluator: {e}")
    Evaluator = None

try:
    from src.fine_tuner import FineTuner
except ImportError as e:
    print(f"Warning: Could not import FineTuner: {e}")
    FineTuner = None

try:
    from src.tutor import AITutor
except ImportError as e:
    print(f"Warning: Could not import AITutor: {e}")
    AITutor = None

try:
    from src.crowdsource import CrowdsourceManager
except ImportError as e:
    print(f"Warning: Could not import CrowdsourceManager: {e}")
    CrowdsourceManager = None

try:
    from src.voice import VoiceProcessor
except ImportError as e:
    print(f"Warning: Could not import VoiceProcessor: {e}")
    VoiceProcessor = None

try:
    from src.fusion import ModelFusion
except ImportError as e:
    print(f"Warning: Could not import ModelFusion: {e}")
    ModelFusion = None

try:
    from src.collaboration import CollaborationArena
except ImportError as e:
    print(f"Warning: Could not import CollaborationArena: {e}")
    CollaborationArena = None

try:
    from src.ethics import EthicsAnalyzer
except ImportError as e:
    print(f"Warning: Could not import EthicsAnalyzer: {e}")
    EthicsAnalyzer = None

try:
    from utils.logger import setup_logger
    from utils.visualizer import Visualizer
    from utils.explainability import Explainability
    from utils.helpers import validate_text, validate_model_params, ensure_directory
except ImportError as e:
    print(f"Warning: Could not import utilities: {e}")
    setup_logger = Visualizer = Explainability = None

try:
    from src.models.llama_model import LlamaModel
    from src.models.openai_model import OpenAIModel
except ImportError as e:
    print(f"Warning: Could not import model classes: {e}")
    LlamaModel = OpenAIModel = None

# Import document processing libraries
try:
    import PyPDF2
    import pytesseract
    from PIL import Image
    import io
    PDF_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Document processing not available: {e}")
    PDF_PROCESSING_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Workbench API",
    description="A comprehensive AI platform for text processing, translation, and chat",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for components
logger = None
config = None
models = []
summarizer = None
translator = None
chatter = None
evaluator = None
retriever = None
visualizer = None
fine_tuner = None
tutor = None
crowdsource = None
voice_processor = None
fusion = None
collaboration = None
ethics = None
explainability = None

def initialize_components():
    """Initialize all components with error handling"""
    global logger, config, models, summarizer, translator, chatter
    global evaluator, retriever, visualizer, fine_tuner, tutor
    global crowdsource, voice_processor, fusion, collaboration, ethics, explainability
    
    try:
        # Setup logger
        if setup_logger:
            logger = setup_logger(__name__)
            logger.info("🚀 Initializing AI Workbench API...")
        else:
            print("⚠️ Logger not available, using print statements")
        
        # Load configuration
        try:
            with open("config/config.yaml", 'r') as file:
                config = yaml.safe_load(file)
            log_info("✓ Configuration loaded")
        except FileNotFoundError:
            log_error("❌ config/config.yaml not found")
            raise HTTPException(status_code=500, detail="Configuration file not found")
        except yaml.YAMLError as e:
            log_error(f"❌ Error parsing config.yaml: {e}")
            raise HTTPException(status_code=500, detail="Configuration file error")
        
        # Ensure directories exist
        ensure_directory("data")
        ensure_directory("data/cache")
        ensure_directory("data/crowdsourced")
        ensure_directory("logs")
        ensure_directory("chroma_db")
        
        # Initialize models
        models = []
        model_errors = []
        
        # OpenAI Model (priority)
        if OpenAIModel and config.get("models", {}).get("openai", {}).get("enabled", True):
            try:
                openai_model = OpenAIModel(config["models"]["openai"])
                if openai_model.is_available():
                    models.append(openai_model)
                    log_info("✓ OpenAI model initialized")
                else:
                    model_errors.append("OpenAI model not available (API connection failed)")
            except Exception as e:
                model_errors.append(f"OpenAI model error: {str(e)}")
        
        # LLaMA Model (optional)
        if LlamaModel and config.get("models", {}).get("llama", {}).get("enabled", False):
            try:
                llama_model = LlamaModel(config["models"]["llama"])
                if llama_model.is_available():
                    models.append(llama_model)
                    log_info("✓ LLaMA model initialized")
                else:
                    log_info("ℹ️ LLaMA model disabled or not available")
            except Exception as e:
                log_warning(f"LLaMA model not available: {str(e)}")
        
        if not models:
            error_msg = "No models available. " + "; ".join(model_errors)
            log_error(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        log_info(f"✓ Initialized {len(models)} model(s): {[m.get_name() for m in models]}")
        
        # Initialize task components
        try:
            if Summarizer:
                summarizer = Summarizer(models)
                log_info("✓ Summarizer initialized")
        except Exception as e:
            log_warning(f"Summarizer initialization failed: {e}")
        
        try:
            if Translator:
                translator = Translator(models)
                log_info("✓ Translator initialized")
        except Exception as e:
            log_warning(f"Translator initialization failed: {e}")
        
        try:
            if Chatter:
                chatter = Chatter(models)
                log_info("✓ Chatter initialized")
        except Exception as e:
            log_warning(f"Chatter initialization failed: {e}")
        
        # Initialize other components (optional)
        try:
            if Evaluator:
                evaluator = Evaluator()
                log_info("✓ Evaluator initialized")
        except Exception as e:
            log_warning(f"Evaluator initialization failed: {e}")
        
        try:
            if RAGRetriever:
                retriever = RAGRetriever()
                log_info("✓ RAG Retriever initialized")
        except Exception as e:
            log_warning(f"RAG Retriever initialization failed: {e}")
        
        try:
            if Visualizer:
                visualizer = Visualizer()
                log_info("✓ Visualizer initialized")
        except Exception as e:
            log_warning(f"Visualizer initialization failed: {e}")
        
        try:
            if FineTuner and config.get("fine_tuning", {}).get("enabled", False):
                fine_tuner = FineTuner()
                log_info("✓ Fine Tuner initialized")
        except Exception as e:
            log_warning(f"Fine Tuner initialization failed: {e}")
        
        try:
            if AITutor:
                tutor = AITutor()
                log_info("✓ AI Tutor initialized")
        except Exception as e:
            log_warning(f"AI Tutor initialization failed: {e}")
        
        try:
            if CrowdsourceManager and config.get("crowdsourcing", {}).get("enabled", True):
                crowdsource = CrowdsourceManager()
                log_info("✓ Crowdsource Manager initialized")
        except Exception as e:
            log_warning(f"Crowdsource Manager initialization failed: {e}")
        
        try:
            if VoiceProcessor and config.get("voice", {}).get("input_enabled", True):
                voice_processor = VoiceProcessor()
                log_info("✓ Voice Processor initialized")
        except Exception as e:
            log_warning(f"Voice Processor initialization failed: {e}")
        
        try:
            if ModelFusion:
                fusion = ModelFusion()
                log_info("✓ Model Fusion initialized")
        except Exception as e:
            log_warning(f"Model Fusion initialization failed: {e}")
        
        try:
            if CollaborationArena:
                collaboration = CollaborationArena()
                log_info("✓ Collaboration Arena initialized")
        except Exception as e:
            log_warning(f"Collaboration Arena initialization failed: {e}")
        
        try:
            if EthicsAnalyzer and config.get("ethics", {}).get("enabled", True):
                ethics = EthicsAnalyzer()
                log_info("✓ Ethics Analyzer initialized")
        except Exception as e:
            log_warning(f"Ethics Analyzer initialization failed: {e}")
        
        try:
            if Explainability:
                explainability = Explainability()
                log_info("✓ Explainability initialized")
        except Exception as e:
            log_warning(f"Explainability initialization failed: {e}")
        
        log_info("🌟 AI Workbench API initialization complete!")
        
    except Exception as e:
        log_error(f"❌ Critical initialization error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize API: {str(e)}")

def log_info(message: str):
    """Log info message"""
    if logger:
        logger.info(message)
    else:
        print(f"INFO: {message}")

def log_warning(message: str):
    """Log warning message"""
    if logger:
        logger.warning(message)
    else:
        print(f"WARNING: {message}")

def log_error(message: str):
    """Log error message"""
    if logger:
        logger.error(message)
    else:
        print(f"ERROR: {message}")

# Pydantic models for request validation
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

# Initialize components on startup
@app.on_event("startup")
async def startup_event():
    """Initialize components when the API starts"""
    initialize_components()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        available_models = [model.get_name() for model in models if model.is_available()] if models else []
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "models_available": len(available_models),
            "model_names": available_models,
            "components": {
                "summarizer": summarizer is not None,
                "translator": translator is not None,
                "chatter": chatter is not None,
                "evaluator": evaluator is not None,
                "retriever": retriever is not None,
                "voice_processor": voice_processor is not None,
                "ethics": ethics is not None
            }
        }
    except Exception as e:
        log_error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main processing endpoint
@app.post("/process")
async def process_task(input: TaskInput):
    """Main task processing endpoint"""
    try:
        log_info(f"Processing request: task={input.task}")
        
        # Validate task type
        if input.task not in ["summarization", "translation", "chat"]:
            raise HTTPException(status_code=400, detail=f"Unsupported task: {input.task}")
        
        # Validate parameters
        try:
            params = validate_model_params(input.params) if validate_model_params else input.params
        except Exception as e:
            log_warning(f"Parameter validation failed: {e}")
            params = input.params
        
        # Retrieve context if available
        context = []
        if retriever and input.text:
            try:
                context = retriever.retrieve(input.text)
                log_info(f"Retrieved {len(context)} context documents")
            except Exception as e:
                log_warning(f"Context retrieval failed: {e}")
        
        # Process based on task type
        if input.task == "summarization":
            if not summarizer:
                raise HTTPException(status_code=503, detail="Summarizer not available")
            
            if not input.text:
                raise HTTPException(status_code=400, detail="Text is required for summarization")
            
            # Add context to text
            text_with_context = input.text
            if context:
                context_text = "\n".join([doc.get("text", "") for doc in context])
                text_with_context += f"\n\nContext: {context_text}"
            
            results = summarizer.summarize(text_with_context, params)
            
            # Evaluate if reference provided
            evaluation = None
            if evaluator and input.reference:
                try:
                    evaluation = evaluator.evaluate_summarization(results, input.reference)
                    evaluation = evaluation.to_dict(orient="records") if evaluation is not None else None
                except Exception as e:
                    log_warning(f"Evaluation failed: {e}")
            
            return {
                "results": results,
                "evaluation": evaluation,
                "context_used": len(context) > 0,
                "task": input.task
            }
        
        elif input.task == "translation":
            if not translator:
                raise HTTPException(status_code=503, detail="Translator not available")
            
            if not input.text or not input.target_lang:
                raise HTTPException(status_code=400, detail="Text and target language are required for translation")
            
            # Add context to text
            text_with_context = input.text
            if context:
                context_text = "\n".join([doc.get("text", "") for doc in context])
                text_with_context += f"\n\nContext: {context_text}"
            
            results = translator.translate(text_with_context, input.target_lang, params)
            
            # Evaluate if reference provided
            evaluation = None
            if evaluator and input.reference:
                try:
                    evaluation = evaluator.evaluate_translation(results, input.reference)
                    evaluation = evaluation.to_dict(orient="records") if evaluation is not None else None
                except Exception as e:
                    log_warning(f"Evaluation failed: {e}")
            
            return {
                "results": results,
                "evaluation": evaluation,
                "context_used": len(context) > 0,
                "target_language": input.target_lang,
                "task": input.task
            }
        
        elif input.task == "chat":
            if not chatter:
                raise HTTPException(status_code=503, detail="Chatter not available")
            
            # Prepare messages
            messages = input.messages or []
            if input.text and not messages:
                messages = [{"role": "user", "content": input.text}]
            
            if not messages:
                raise HTTPException(status_code=400, detail="Messages are required for chat")
            
            # Add context to the last user message if available
            if context and messages:
                last_message = messages[-1]
                if last_message.get("role") == "user":
                    context_text = "\n".join([doc.get("text", "") for doc in context])
                    last_message["content"] += f"\n\nRelevant context: {context_text}"
            
            results = chatter.chat(messages, params)
            
            # For chat, return just the response from the first available model
            if results and results[0].get("output"):
                return results[0]["output"]
            else:
                raise HTTPException(status_code=500, detail="No successful chat response generated")
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document upload endpoint
@app.post("/upload_documents")
async def upload_documents(file: UploadFile = File(...)):
    """Upload and process documents for RAG"""
    try:
        if not retriever:
            raise HTTPException(status_code=503, detail="Document processing not available")
        
        if not PDF_PROCESSING_AVAILABLE:
            raise HTTPException(status_code=503, detail="Document processing libraries not available")
        
        # Process based on file type
        content = await file.read()
        
        if file.filename.endswith(".pdf"):
            try:
                pdf = PyPDF2.PdfReader(io.BytesIO(content))
                text = "".join([page.extract_text() for page in pdf.pages])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(e)}")
        
        elif file.filename.endswith((".png", ".jpg", ".jpeg")):
            try:
                image = Image.open(io.BytesIO(content))
                text = pytesseract.image_to_string(image)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or image files.")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from document")
        
        # Add to retriever
        retriever.add_documents([text])
        
        log_info(f"Document uploaded: {file.filename}, extracted {len(text)} characters")
        
        return {
            "message": "Document uploaded and indexed successfully",
            "filename": file.filename,
            "text_length": len(text),
            "word_count": len(text.split())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Voice input endpoint
@app.post("/voice_input")
async def voice_input(file: UploadFile = File(...)):
    """Process voice input"""
    try:
        if not voice_processor:
            raise HTTPException(status_code=503, detail="Voice processing not available")
        
        audio_data = await file.read()
        text = voice_processor.speech_to_text(audio_data)
        
        return {"text": text}
        
    except Exception as e:
        log_error(f"Voice input error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Voice output endpoint
@app.post("/voice_output")
async def voice_output(text: str):
    """Generate voice output"""
    try:
        if not voice_processor:
            raise HTTPException(status_code=503, detail="Voice processing not available")
        
        output_path = voice_processor.text_to_speech(text)
        return {"audio_path": output_path}
        
    except Exception as e:
        log_error(f"Voice output error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Crowdsourcing endpoints
@app.post("/crowdsource")
async def crowdsource_dataset(input: CrowdsourceInput):
    """Submit dataset for crowdsourcing"""
    try:
        if not crowdsource:
            raise HTTPException(status_code=503, detail="Crowdsourcing not available")
        
        dataset_id = crowdsource.submit_dataset(input.data, input.submitter)
        return {"dataset_id": dataset_id, "message": "Dataset submitted for review"}
        
    except Exception as e:
        log_error(f"Crowdsource submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pending_datasets")
async def get_pending_datasets():
    """Get pending datasets for moderation"""
    try:
        if not crowdsource:
            raise HTTPException(status_code=503, detail="Crowdsourcing not available")
        
        datasets = crowdsource.get_pending_datasets()
        return {"datasets": datasets}
        
    except Exception as e:
        log_error(f"Get pending datasets error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/approve_dataset/{dataset_id}")
async def approve_dataset(dataset_id: int):
    """Approve a crowdsourced dataset"""
    try:
        if not crowdsource:
            raise HTTPException(status_code=503, detail="Crowdsourcing not available")
        
        crowdsource.approve_dataset(dataset_id)
        return {"message": f"Dataset {dataset_id} approved"}
        
    except Exception as e:
        log_error(f"Dataset approval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for collaboration
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time collaboration"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            try:
                if collaboration:
                    await collaboration.broadcast_results(data.get("task", "unknown"), data.get("results", []))
                await websocket.send_json({"status": "received", "timestamp": time.time()})
            except Exception as e:
                log_error(f"WebSocket processing error: {e}")
                await websocket.send_json({"status": "error", "message": str(e)})
    except WebSocketDisconnect:
        log_info("WebSocket client disconnected")
    except Exception as e:
        log_error(f"WebSocket error: {e}")

# Additional utility endpoints
@app.get("/models")
async def get_models():
    """Get information about available models"""
    try:
        model_info = []
        for model in models:
            info = model.get_model_info() if hasattr(model, 'get_model_info') else {
                "name": model.get_name(),
                "available": model.is_available()
            }
            model_info.append(info)
        
        return {"models": model_info}
        
    except Exception as e:
        log_error(f"Get models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported_languages")
async def get_supported_languages():
    """Get supported languages for translation"""
    try:
        if translator and hasattr(translator, 'get_supported_languages'):
            return {"languages": translator.get_supported_languages()}
        else:
            # Default languages
            return {"languages": ["Spanish", "French", "German", "Italian", "Portuguese", "Chinese", "Japanese", "Korean"]}
    except Exception as e:
        log_error(f"Get supported languages error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/usage_stats")
async def get_usage_stats():
    """Get usage statistics"""
    try:
        stats = {}
        for model in models:
            if hasattr(model, 'get_usage_stats'):
                model_stats = model.get_usage_stats()
                stats[model.get_name()] = model_stats
        
        return {"usage_stats": stats}
        
    except Exception as e:
        log_error(f"Get usage stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    log_error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )