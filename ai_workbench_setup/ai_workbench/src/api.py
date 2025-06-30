from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, ValidationError
import os
import yaml
import json
import time
import threading
from utils.helpers import clean_response_text
import tempfile
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
            logger.info("🚀 Initializing AI Workbench API with Voice Capabilities...")
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
        ensure_directory("data/voice_output")  # Voice output directory
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
        
        # Initialize Voice Processor (Enhanced for ChatGPT-like capabilities)
        try:
            if VoiceProcessor and config.get("voice", {}).get("input_enabled", True):
                voice_processor = VoiceProcessor()
                log_info("✓ Voice Processor initialized with advanced capabilities")
                
                # Test voice system
                voice_test = voice_processor.test_voice_system()
                if voice_test.get("input_available") or voice_test.get("output_available"):
                    log_info("🎤 Voice system ready for ChatGPT-like interaction")
                else:
                    log_warning("⚠️ Voice system has limited functionality")
                    
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
        
        log_info("🌟 AI Workbench API with Voice Capabilities initialization complete!")
        
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

# Lifespan event handler (replaces @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_components()
    yield
    # Shutdown (if needed)
    log_info("🛑 AI Workbench API shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AI Workbench API with Voice",
    description="A comprehensive AI platform with ChatGPT-like voice capabilities",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class VoiceMessage(BaseModel):
    text: str
    messages: Optional[List[Dict]] = None
    params: Dict[str, Any] = {}

# NEW: Voice-specific models
class TextToSpeechInput(BaseModel):
    text: str
    language: str = "en"
    speed: float = 1.0
    pitch: float = 1.0
    voice_type: str = "default"

class VoiceSettings(BaseModel):
    language: str = "en"
    auto_play: bool = True
    continuous_mode: bool = False
    voice_response_enabled: bool = True

# Health check endpoint (Enhanced with voice status)
@app.get("/health")
async def health_check():
    """Health check endpoint with voice system status"""
    try:
        available_models = [model.get_name() for model in models if model.is_available()] if models else []
        
        # Voice system status
        voice_status = {
            "available": voice_processor is not None,
            "input_enabled": False,
            "output_enabled": False,
            "languages_supported": []
        }
        
        if voice_processor:
            try:
                voice_test = voice_processor.test_voice_system()
                voice_status.update({
                    "input_enabled": voice_test.get("input_available", False),
                    "output_enabled": voice_test.get("output_available", False),
                    "microphones_detected": len(voice_test.get("microphones", {})),
                    "languages_supported": ["en", "es", "fr", "de", "it", "pt"]
                })
            except Exception as e:
                log_warning(f"Voice system test failed: {e}")
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "models_available": len(available_models),
            "model_names": available_models,
            "voice_system": voice_status,
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
    """Main task processing endpoint - FIXED: Only use RAG for chat + Proper evaluation for graphs"""
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
        
        # FIXED: Only retrieve context for chat tasks
        context = []
        if input.task == "chat" and retriever:
            # Determine query text for context retrieval
            query_text = ""
            if input.text:
                query_text = input.text
            elif input.messages and len(input.messages) > 0:
                # For chat, use the last user message
                last_msg = input.messages[-1]
                if last_msg.get("role") == "user":
                    query_text = last_msg.get("content", "")
            
            # Retrieve context if we have query text
            if query_text:
                try:
                    context = retriever.retrieve(query_text)
                    log_info(f"Retrieved {len(context)} context documents for chat query: {query_text[:50]}...")
                except Exception as e:
                    log_warning(f"Context retrieval failed: {e}")
        
        # Process based on task type
        if input.task == "summarization":
            if not summarizer:
                raise HTTPException(status_code=503, detail="Summarizer not available")
            
            if not input.text:
                raise HTTPException(status_code=400, detail="Text is required for summarization")
            
            # FIXED: No context for summarization - use original text only
            results = summarizer.summarize(input.text, params)
            
            # FIXED: Ensure evaluation for graphs
            evaluation = None
            if evaluator and input.reference:
                try:
                    # Create evaluation DataFrame for visualization
                    evaluation = evaluator.evaluate_summarization(results, input.reference)
                    if evaluation is not None:
                        evaluation = evaluation.to_dict(orient="records")
                    log_info("✓ Summarization evaluation completed for graphs")
                except Exception as e:
                    log_warning(f"Evaluation failed: {e}")
            
            # FIXED: Return proper format for graphs
            return {
                "results": results,
                "evaluation": evaluation,
                "context_used": False,  # Never use context for summarization
                "task": input.task,
                "has_evaluation": evaluation is not None
            }
        
        elif input.task == "translation":
            if not translator:
                raise HTTPException(status_code=503, detail="Translator not available")
            
            if not input.text or not input.target_lang:
                raise HTTPException(status_code=400, detail="Text and target language are required for translation")
            
            # FIXED: No context for translation - use original text only
            results = translator.translate(input.text, input.target_lang, params)
            
            # FIXED: Ensure evaluation for graphs
            evaluation = None
            if evaluator and input.reference:
                try:
                    evaluation = evaluator.evaluate_translation(results, input.reference)
                    if evaluation is not None:
                        evaluation = evaluation.to_dict(orient="records")
                    log_info("✓ Translation evaluation completed for graphs")
                except Exception as e:
                    log_warning(f"Evaluation failed: {e}")
            
            # FIXED: Return proper format for graphs
            return {
                "results": results,
                "evaluation": evaluation,
                "context_used": False,  # Never use context for translation
                "target_language": input.target_lang,
                "task": input.task,
                "has_evaluation": evaluation is not None
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
            
            # FIXED: Only add context to chat messages
            if context and messages:
                last_message = messages[-1]
                if last_message.get("role") == "user":
                    context_text = "\n".join([doc.get("text", "") for doc in context])
                    # Add context in a more natural way
                    last_message["content"] += f"\n\nRelevant information from uploaded documents:\n{context_text}"
                    log_info(f"Added context from {len(context)} documents to chat message")
            
            results = chatter.chat(messages, params)
            
            # For chat, return just the response from the first available model
            if results and results[0].get("output"):
                return clean_response_text(results[0]["output"])
            else:
                raise HTTPException(status_code=500, detail="No successful chat response generated")
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ENHANCED Voice chat endpoint with automatic audio response
@app.post("/voice_chat")
async def voice_chat(voice_input: VoiceMessage):
    """Enhanced voice chat endpoint with automatic audio response generation"""
    try:
        if not voice_processor:
            raise HTTPException(status_code=503, detail="Voice processing not available")
        
        if not chatter:
            raise HTTPException(status_code=503, detail="Chat not available")
        
        log_info(f"Processing voice chat: {voice_input.text[:50]}...")
        
        # Prepare messages
        messages = voice_input.messages or []
        if voice_input.text:
            messages.append({"role": "user", "content": voice_input.text})
        
        if not messages:
            raise HTTPException(status_code=400, detail="No voice input provided")
        
        # Get context for voice chat (same as regular chat)
        context = []
        if retriever and messages:
            query_text = messages[-1].get("content", "")
            if query_text:
                try:
                    context = retriever.retrieve(query_text)
                    log_info(f"Retrieved {len(context)} context documents for voice chat")
                except Exception as e:
                    log_warning(f"Voice chat context retrieval failed: {e}")
        
        # Add context if available
        if context and messages:
            last_message = messages[-1]
            if last_message.get("role") == "user":
                context_text = "\n".join([doc.get("text", "") for doc in context])
                last_message["content"] += f"\n\nRelevant information:\n{context_text}"
        
        # Get chat response
        results = chatter.chat(messages, voice_input.params)
        
        if results and results[0].get("output"):
            response_text = results[0]["output"]
            
            # Generate speech audio automatically
            audio_path = None
            audio_url = None
            try:
                audio_path = voice_processor.text_to_speech(
                    text=response_text,
                    language=voice_input.params.get("language", "en")
                )
                
                if audio_path:
                    # Create accessible URL for the audio file
                    audio_filename = os.path.basename(audio_path)
                    audio_url = f"/audio/{audio_filename}"
                    log_info("✓ Voice response generated successfully")
                    
            except Exception as e:
                log_warning(f"Text-to-speech failed: {e}")
                # Continue without audio - don't fail the whole request
            
            return {
                "text": clean_response_text(response_text),
                "audio_path": audio_path,
                "audio_url": audio_url,
                "audio_available": audio_path is not None,
                "context_used": len(context) > 0,
                "inference_time": results[0].get("inference_time", 0),
                "model_used": results[0].get("model", "unknown")
            }
        else:
            raise HTTPException(status_code=500, detail="No successful chat response generated")
            
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Speech-to-text endpoint (Enhanced)
@app.post("/speech_to_text")
async def speech_to_text(file: UploadFile = File(...)):
    """Convert speech to text with enhanced processing"""
    try:
        if not voice_processor:
            raise HTTPException(status_code=503, detail="Voice processing not available")
        
        log_info(f"Processing speech-to-text for file: {file.filename}")
        
        # Read audio data
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Convert to text with enhanced processing
        start_time = time.time()
        text = voice_processor.speech_to_text(audio_data)
        processing_time = time.time() - start_time
        
        log_info(f"Speech-to-text result: {text[:50]}..." if text else "No speech detected")
        
        return {
            "text": text,
            "processing_time": processing_time,
            "file_size": len(audio_data),
            "success": bool(text and text.strip()),
            "language_detected": "auto"  # Could be enhanced with language detection
        }
        
    except Exception as e:
        log_error(f"Speech-to-text error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Text-to-speech endpoint
@app.post("/text_to_speech")
async def text_to_speech_endpoint(input: TextToSpeechInput):
    """
    Convert text to speech and return audio file
    
    Args:
        input: Text-to-speech parameters
        
    Returns:
        Audio file response
    """
    try:
        if not voice_processor:
            raise HTTPException(status_code=503, detail="Voice processing not available")
        
        log_info(f"Generating speech for text: {input.text[:50]}...")
        
        # Validate input
        if not input.text or not input.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(input.text) > 5000:  # Limit text length
            raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
        
        # Generate speech audio file
        audio_path = voice_processor.text_to_speech(
            text=input.text,
            language=input.language,
            slow=(input.speed < 1.0)
        )
        
        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        log_info(f"✓ Speech generated successfully: {audio_path}")
        
        # Return the audio file
        return FileResponse(
            path=audio_path,
            media_type="audio/mpeg",
            filename=f"speech_output_{int(time.time())}.mp3",
            headers={
                "Content-Disposition": "attachment; filename=speech_output.mp3",
                "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Text-to-speech error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Streaming text-to-speech endpoint
@app.post("/text_to_speech_stream")
async def text_to_speech_stream(input: TextToSpeechInput):
    """
    Convert text to speech and return streaming audio
    """
    try:
        if not voice_processor:
            raise HTTPException(status_code=503, detail="Voice processing not available")
        
        # Generate speech
        audio_path = voice_processor.text_to_speech(
            text=input.text,
            language=input.language,
            slow=(input.speed < 1.0)
        )
        
        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # Stream the audio file
        def iterfile(file_path: str):
            with open(file_path, mode="rb") as file_like:
                yield from file_like
        
        # Clean up the file after streaming
        def cleanup():
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    log_info(f"Cleaned up temporary audio file: {audio_path}")
            except Exception as e:
                log_warning(f"Could not clean up audio file {audio_path}: {e}")
        
        response = StreamingResponse(
            iterfile(audio_path),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=speech.mp3",
                "Cache-Control": "no-cache"
            }
        )
        
        # Schedule cleanup after 10 seconds
        threading.Timer(10.0, cleanup).start()
        
        return response
        
    except Exception as e:
        log_error(f"Text-to-speech streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Serve audio files
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """
    Serve audio files generated by the system
    """
    try:
        # Construct safe file path
        audio_dir = Path("data/voice_output")
        file_path = audio_dir / filename
        
        # Security check - ensure file is in the correct directory
        if not str(file_path.resolve()).startswith(str(audio_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Get file info
        file_size = file_path.stat().st_size
        
        return FileResponse(
            path=str(file_path),
            media_type="audio/mpeg",
            filename=filename,
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Error serving audio file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Voice system status
@app.get("/voice_status")
async def get_voice_status():
    """
    Get voice system status and capabilities
    """
    try:
        status = {
            "voice_system_available": voice_processor is not None,
            "voice_input_enabled": False,
            "voice_output_enabled": False,
            "supported_languages": ["en", "es", "fr", "de", "it", "pt"],
            "supported_formats": ["wav", "mp3", "webm", "ogg"],
            "max_text_length": 5000,
            "voice_processor_loaded": voice_processor is not None,
            "features": {
                "speech_to_text": voice_processor is not None,
                "text_to_speech": voice_processor is not None,
                "streaming_audio": voice_processor is not None,
                "multi_language": True,
                "noise_cancellation": True,
                "auto_gain_control": True
            }
        }
        
        if voice_processor:
            # Test voice system if available
            try:
                test_results = voice_processor.test_voice_system()
                status.update({
                    "voice_input_enabled": test_results.get("input_available", False),
                    "voice_output_enabled": test_results.get("output_available", False),
                    "microphones_detected": test_results.get("microphones", {}),
                    "test_results": test_results,
                    "last_test": time.time()
                })
            except Exception as e:
                log_warning(f"Voice system test failed: {e}")
                status["test_error"] = str(e)
        
        return status
        
    except Exception as e:
        log_error(f"Error getting voice status: {e}")
        return {"error": str(e), "voice_available": False}

# NEW: Voice settings management
@app.post("/voice_settings")
async def update_voice_settings(settings: VoiceSettings):
    """
    Update voice system settings
    """
    try:
        # Store settings in global config (in production, use database)
        if not hasattr(app.state, "voice_settings"):
            app.state.voice_settings = {}
        
        app.state.voice_settings.update({
            "language": settings.language,
            "auto_play": settings.auto_play,
            "continuous_mode": settings.continuous_mode,
            "voice_response_enabled": settings.voice_response_enabled,
            "updated_at": time.time()
        })
        
        log_info(f"Voice settings updated: {settings.dict()}")
        
        return {
            "success": True,
            "settings": settings.dict(),
            "message": "Voice settings updated successfully"
        }
        
    except Exception as e:
        log_error(f"Error updating voice settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voice_settings")
async def get_voice_settings():
    """
    Get current voice settings
    """
    try:
        default_settings = {
            "language": "en",
            "auto_play": True,
            "continuous_mode": False,
            "voice_response_enabled": True
        }
        
        if hasattr(app.state, "voice_settings"):
            return app.state.voice_settings
        else:
            return default_settings
            
    except Exception as e:
        log_error(f"Error getting voice settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Audio cleanup endpoint
@app.post("/cleanup_audio")
async def cleanup_old_audio():
    """
    Clean up old audio files to save disk space
    """
    try:
        audio_dir = Path("data/voice_output")
        if not audio_dir.exists():
            return {"message": "No audio directory found", "deleted_count": 0}
        
        # Delete files older than 1 hour
        current_time = time.time()
        deleted_count = 0
        total_size_deleted = 0
        
        for file_path in audio_dir.glob("*.mp3"):
            try:
                file_age = current_time - file_path.stat().st_mtime
                if file_age > 3600:  # 1 hour
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    deleted_count += 1
                    total_size_deleted += file_size
            except Exception as e:
                log_warning(f"Could not delete {file_path}: {e}")
        
        log_info(f"Cleaned up {deleted_count} old audio files ({total_size_deleted / 1024 / 1024:.2f} MB)")
        return {
            "message": f"Cleaned up {deleted_count} old audio files",
            "deleted_count": deleted_count,
            "size_freed_mb": round(total_size_deleted / 1024 / 1024, 2)
        }
        
    except Exception as e:
        log_error(f"Audio cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document upload endpoint (existing, kept as is)
@app.post("/upload_documents")
async def upload_documents(file: UploadFile = File(...)):
    """Upload and process documents for RAG"""
    try:
        if not retriever:
            raise HTTPException(status_code=503, detail="Document processing not available")
        
        if not PDF_PROCESSING_AVAILABLE:
            raise HTTPException(status_code=503, detail="Document processing libraries not available")
        
        log_info(f"Processing document upload: {file.filename}")
        
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

# Crowdsourcing endpoints (existing, kept as is)
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

# WebSocket endpoint for collaboration (existing, kept as is)
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

# Additional utility endpoints (existing, kept as is)
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
    """Get supported languages for translation and voice"""
    try:
        translation_languages = ["Spanish", "French", "German", "Italian", "Portuguese", "Chinese", "Japanese", "Korean"]
        voice_languages = ["en", "es", "fr", "de", "it", "pt"]
        
        if translator and hasattr(translator, 'get_supported_languages'):
            translation_languages = translator.get_supported_languages()
        
        return {
            "translation_languages": translation_languages,
            "voice_languages": voice_languages,
            "voice_language_names": {
                "en": "English",
                "es": "Spanish", 
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese"
            }
        }
    except Exception as e:
        log_error(f"Get supported languages error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/usage_stats")
async def get_usage_stats():
    """Get usage statistics including voice usage"""
    try:
        stats = {"voice_usage": {}}
        
        # Model usage stats
        for model in models:
            if hasattr(model, 'get_usage_stats'):
                model_stats = model.get_usage_stats()
                stats[model.get_name()] = model_stats
        
        # Voice usage stats (basic implementation)
        if voice_processor:
            audio_dir = Path("data/voice_output")
            if audio_dir.exists():
                audio_files = list(audio_dir.glob("*.mp3"))
                total_size = sum(f.stat().st_size for f in audio_files if f.exists())
                stats["voice_usage"] = {
                    "total_audio_files": len(audio_files),
                    "total_size_mb": round(total_size / 1024 / 1024, 2),
                    "voice_processor_available": True
                }
        
        return {"usage_stats": stats}
        
    except Exception as e:
        log_error(f"Get usage stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Voice conversation history endpoint
@app.get("/voice_conversations")
async def get_voice_conversations(limit: int = 10):
    """Get recent voice conversation history"""
    try:
        # This is a basic implementation - in production you'd use a database
        conversations = []
        
        # Get recent chat conversations if chatter has history
        if chatter and hasattr(chatter, 'get_conversation_history'):
            conversations = chatter.get_conversation_history(limit=limit)
        
        # Add voice-specific metadata
        for conv in conversations:
            conv["has_voice"] = True
            conv["voice_enabled"] = voice_processor is not None
        
        return {
            "conversations": conversations,
            "total": len(conversations),
            "voice_system_available": voice_processor is not None
        }
        
    except Exception as e:
        log_error(f"Error getting voice conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Voice analytics endpoint
@app.get("/voice_analytics")
async def get_voice_analytics():
    """Get voice system analytics and performance metrics"""
    try:
        analytics = {
            "system_status": voice_processor is not None,
            "uptime": time.time(),  # Simplified
            "total_sessions": 0,
            "avg_response_time": 0.0,
            "languages_used": {},
            "error_rate": 0.0,
            "performance_metrics": {
                "speech_to_text_accuracy": "95%",  # Mock data
                "text_to_speech_quality": "High",
                "latency_ms": 150,
                "success_rate": "98%"
            }
        }
        
        # Get audio file statistics
        audio_dir = Path("data/voice_output")
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.mp3"))
            analytics["total_audio_generated"] = len(audio_files)
            analytics["storage_used_mb"] = round(
                sum(f.stat().st_size for f in audio_files if f.exists()) / 1024 / 1024, 2
            )
        
        return analytics
        
    except Exception as e:
        log_error(f"Error getting voice analytics: {e}")
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

# Startup message
if __name__ == "__main__":
    log_info("🎤 AI Workbench API with ChatGPT-like Voice Capabilities")
    log_info("🚀 Ready to serve requests with advanced voice features!")
    log_info("📚 API Documentation available at: /docs")
    log_info("🔊 Voice endpoints: /voice_chat, /speech_to_text, /text_to_speech")
    log_info("🎯 Voice status: /voice_status")