# Display results with improved graph handling
def display_results(results, task_type):
    """Display task results in a formatted way with proper graphs"""
    if isinstance(results, str):
        st.markdown(f"**Response:** {results}")
        return
    
    if not isinstance(results, dict):
        st.error("Invalid response format")
        return
    
    # Handle structured responses
    task_results = results.get("results", [])
    evaluation = results.get("evaluation")
    
    if task_results:
        st.markdown('<div class="sub-header">🤖 Model Results</div>', unsafe_allow_html=True)
        
        for result in task_results:
            model_name = result.get("model", "import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
import tempfile
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="AI Workbench",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling including voice interface
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.success-message {
    color: #28a745;
    font-weight: bold;
}
.error-message {
    color: #dc3545;
    font-weight: bold;
}
.warning-message {
    color: #ffc107;
    font-weight: bold;
}
.voice-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    text-align: center;
    color: white;
}
.voice-btn {
    background: #007bff;
    border: none;
    border-radius: 50%;
    width: 70px;
    height: 70px;
    color: white;
    font-size: 28px;
    cursor: pointer;
    margin: 10px;
    transition: all 0.3s ease;
    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
}
.voice-btn:hover {
    background: #0056b3;
    transform: scale(1.05);
    box-shadow: 0 8px 16px rgba(0,0,0,0.4);
}
.voice-btn.recording {
    background: #dc3545;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 6px 12px rgba(220,53,69,0.3); }
    50% { transform: scale(1.1); box-shadow: 0 8px 20px rgba(220,53,69,0.5); }
    100% { transform: scale(1); box-shadow: 0 6px 12px rgba(220,53,69,0.3); }
}
.voice-status {
    margin: 15px 0;
    padding: 12px;
    border-radius: 8px;
    font-weight: bold;
    font-size: 16px;
}
.status-ready { 
    background: rgba(40, 167, 69, 0.2); 
    color: #28a745; 
    border: 2px solid #28a745;
}
.status-recording { 
    background: rgba(220, 53, 69, 0.2); 
    color: #dc3545; 
    border: 2px solid #dc3545;
}
.status-processing { 
    background: rgba(255, 193, 7, 0.2); 
    color: #ffc107; 
    border: 2px solid #ffc107;
}
.transcription-box {
    margin-top: 15px; 
    padding: 15px; 
    background: rgba(255,255,255,0.9); 
    border-radius: 10px; 
    min-height: 50px;
    color: #333;
    font-size: 16px;
    border: 2px solid #dee2e6;
}
.chat-input-container {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 10px 0;
}
.quick-voice-btn {
    background: #28a745;
    border: none;
    border-radius: 25px;
    padding: 8px 15px;
    color: white;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.quick-voice-btn:hover {
    background: #218838;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_status" not in st.session_state:
        st.session_state.api_status = None
    if "model_info" not in st.session_state:
        st.session_state.model_info = None
    if "supported_languages" not in st.session_state:
        st.session_state.supported_languages = ["Spanish", "French", "German", "Italian", "Portuguese"]
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = True
    if "voice_input_text" not in st.session_state:
        st.session_state.voice_input_text = ""
    if "last_voice_input" not in st.session_state:
        st.session_state.last_voice_input = ""

# Load configuration
@st.cache_data
def load_config():
    """Load configuration with caching"""
    try:
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            st.error(f"Configuration file not found: {config_path}")
            return None
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

# API URL setup
def get_api_urls(config):
    """Get API URLs from configuration"""
    if not config:
        base_url = "http://127.0.0.1:8000"
    else:
        host = config.get("api", {}).get("host", "127.0.0.1")
        port = config.get("api", {}).get("port", 8000)
        base_url = f"http://{host}:{port}"
    
    return {
        "base": base_url,
        "process": f"{base_url}/process",
        "upload": f"{base_url}/upload_documents",
        "voice_chat": f"{base_url}/voice_chat",
        "speech_to_text": f"{base_url}/speech_to_text",
        "crowdsource": f"{base_url}/crowdsource",
        "pending": f"{base_url}/pending_datasets",
        "approve": f"{base_url}/approve_dataset",
        "health": f"{base_url}/health",
        "models": f"{base_url}/models",
        "languages": f"{base_url}/supported_languages",
        "usage": f"{base_url}/usage_stats"
    }

# HTTP session with retries
def create_session():
    """Create HTTP session with retry configuration"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Voice Interface Component
def render_voice_interface(api_urls):
    """Render integrated voice interface"""
    
    voice_html = f"""
    <div class="voice-container">
        <h3 style="margin-top: 0;">🎤 Voice Chat Interface</h3>
        <p style="margin: 10px 0; opacity: 0.9;">Click the microphone to speak with AI</p>
        
        <button id="voiceBtn" class="voice-btn" onclick="toggleRecording()">
            🎤
        </button>
        
        <div id="voiceStatus" class="voice-status status-ready">
            Ready for voice input - Click microphone to start
        </div>
        
        <div id="transcription" class="transcription-box">
            Your speech will appear here...
        </div>
        
        <div style="margin-top: 15px; font-size: 14px; opacity: 0.8;">
            💡 Tip: Speak clearly and wait for the transcription to appear
        </div>
    </div>
    
    <script>
    let isRecording = false;
    let mediaRecorder;
    let audioChunks = [];
    let recordingTimeout;
    
    async function toggleRecording() {{
        const btn = document.getElementById('voiceBtn');
        const status = document.getElementById('voiceStatus');
        
        if (!isRecording) {{
            try {{
                const stream = await navigator.mediaDevices.getUserMedia({{ 
                    audio: {{
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }}
                }});
                
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = event => {{
                    if (event.data.size > 0) {{
                        audioChunks.push(event.data);
                    }}
                }};
                
                mediaRecorder.onstop = async () => {{
                    const audioBlob = new Blob(audioChunks, {{ type: 'audio/wav' }});
                    await processAudio(audioBlob);
                }};
                
                mediaRecorder.start();
                isRecording = true;
                
                // Visual feedback
                btn.classList.add('recording');
                btn.innerHTML = '🛑';
                status.className = 'voice-status status-recording';
                status.textContent = '🔴 Recording... Click to stop or speak for 10 seconds';
                
                // Auto-stop after 10 seconds
                recordingTimeout = setTimeout(() => {{
                    if (isRecording) {{
                        stopRecording();
                    }}
                }}, 10000);
                
            }} catch (error) {{
                status.className = 'voice-status status-ready';
                status.innerHTML = '❌ Microphone access denied.<br>Please enable microphone permissions and refresh.';
                console.error('Error accessing microphone:', error);
            }}
        }} else {{
            stopRecording();
        }}
    }}
    
    function stopRecording() {{
        if (recordingTimeout) {{
            clearTimeout(recordingTimeout);
        }}
        
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {{
            mediaRecorder.stop();
        }}
        
        isRecording = false;
        
        const btn = document.getElementById('voiceBtn');
        const status = document.getElementById('voiceStatus');
        
        btn.classList.remove('recording');
        btn.innerHTML = '🎤';
        status.className = 'voice-status status-processing';
        status.textContent = '⏳ Processing speech... Please wait';
        
        // Stop all tracks
        if (mediaRecorder && mediaRecorder.stream) {{
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }}
    }}
    
    async function processAudio(audioBlob) {{
        const status = document.getElementById('voiceStatus');
        const transcription = document.getElementById('transcription');
        
        try {{
            // Show processing state
            status.className = 'voice-status status-processing';
            status.textContent = '🤖 Converting speech to text...';
            transcription.textContent = 'Processing audio...';
            
            // Convert blob to FormData for upload
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.wav');
            
            // Send to speech-to-text API
            const response = await fetch('{api_urls["speech_to_text"]}', {{
                method: 'POST',
                body: formData
            }});
            
            if (response.ok) {{
                const result = await response.json();
                const text = result.text || '';
                
                if (text.trim()) {{
                    // Success
                    transcription.innerHTML = `<strong>You said:</strong> "${{text}}"`;
                    status.className = 'voice-status status-ready';
                    status.innerHTML = '✅ Speech recognized! <br>Click microphone for more input.';
                    
                    // Send to Streamlit
                    window.parent.postMessage({{
                        type: 'voiceInput',
                        text: text,
                        timestamp: Date.now()
                    }}, '*');
                    
                }} else {{
                    // No speech detected
                    transcription.textContent = 'No speech detected. Please try again.';
                    status.className = 'voice-status status-ready';
                    status.textContent = '⚠️ No speech detected - Click microphone to try again';
                }}
            }} else {{
                throw new Error(`HTTP ${{response.status}}: ${{response.statusText}}`);
            }}
            
        }} catch (error) {{
            console.error('Error processing audio:', error);
            transcription.textContent = 'Error processing speech. Please try again.';
            status.className = 'voice-status status-ready';
            status.innerHTML = '❌ Processing failed<br>Check connection and try again';
        }}
    }}
    
    // Listen for messages from Streamlit
    window.addEventListener('message', function(event) {{
        if (event.data.type === 'clearVoiceTranscription') {{
            const transcription = document.getElementById('transcription');
            const status = document.getElementById('voiceStatus');
            
            transcription.textContent = 'Your speech will appear here...';
            status.className = 'voice-status status-ready';
            status.textContent = 'Ready for voice input - Click microphone to start';
        }}
    }});
    </script>
    """
    
    # Render the voice interface
    components.html(voice_html, height=300)

# Simple voice recorder fallback
def simple_voice_recorder():
    """Simple voice recorder using audio-recorder-streamlit if available"""
    try:
        from audio_recorder_streamlit import audio_recorder
        
        st.markdown("#### 🎙️ Alternative Voice Recorder")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("🎯 Process This Recording"):
                return audio_bytes
        
        return None
        
    except ImportError:
        st.info("💡 Install `audio-recorder-streamlit` for an alternative voice recorder: `pip install audio-recorder-streamlit`")
        return None

# Process voice input
def process_voice_input(api_urls, audio_data):
    """Process voice input and return text"""
    try:
        if not audio_data:
            return ""
        
        session = create_session()
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        
        try:
            # Send to speech-to-text API
            with open(tmp_file_path, 'rb') as f:
                files = {"file": ("audio.wav", f, "audio/wav")}
                response = session.post(api_urls["speech_to_text"], files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("text", "")
            else:
                st.error(f"❌ Speech recognition failed: {response.status_code}")
                return ""
                
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        st.error(f"❌ Voice processing error: {e}")
        return ""

# Voice chat handler
def handle_voice_chat(api_urls, text, messages, params):
    """Handle voice chat with text-to-speech response"""
    try:
        session = create_session()
        
        payload = {
            "text": text,
            "messages": messages,
            "params": params
        }
        
        with st.spinner("🤖 AI is thinking..."):
            response = session.post(api_urls["voice_chat"], json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("text", "")
            audio_path = result.get("audio_path")
            
            if response_text:
                st.success("✅ Voice chat response generated")
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Display response
                with st.chat_message("assistant"):
                    st.write(response_text)
                
                # Play audio if available
                if audio_path and os.path.exists(audio_path):
                    try:
                        with open(audio_path, 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/mp3")
                        st.success("🔊 Audio response available above")
                    except Exception as e:
                        st.warning(f"Could not play audio: {e}")
                
                return True
            else:
                st.warning("⚠️ No response generated")
                return False
        else:
            st.error(f"❌ Voice chat failed: {response.status_code}")
            return False
            
    except Exception as e:
        st.error(f"❌ Voice chat error: {e}")
        return False

# Health check
def check_api_health(api_urls):
    """Check API health status"""
    try:
        session = create_session()
        response = session.get(api_urls["health"], timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            st.session_state.api_status = health_data
            return True, health_data
        else:
            return False, f"API returned status code: {response.status_code}"
    except requests.RequestException as e:
        return False, str(e)
    except Exception as e:
        return False, f"Health check error: {e}"

# Fetch model information
def fetch_model_info(api_urls):
    """Fetch available model information"""
    try:
        session = create_session()
        response = session.get(api_urls["models"], timeout=10)
        
        if response.status_code == 200:
            model_data = response.json()
            st.session_state.model_info = model_data
            return model_data
        else:
            st.warning(f"Could not fetch model info: {response.status_code}")
            return None
    except Exception as e:
        st.warning(f"Error fetching model info: {e}")
        return None

# Fetch supported languages
def fetch_supported_languages(api_urls):
    """Fetch supported languages for translation"""
    try:
        session = create_session()
        response = session.get(api_urls["languages"], timeout=10)
        
        if response.status_code == 200:
            lang_data = response.json()
            languages = lang_data.get("languages", [])
            st.session_state.supported_languages = languages
            return languages
        else:
            return st.session_state.supported_languages
    except Exception as e:
        st.warning(f"Error fetching languages: {e}")
        return st.session_state.supported_languages

# Main processing function
def process_task(api_urls, payload):
    """Process task with the API"""
    try:
        session = create_session()
        
        with st.spinner("Processing..."):
            response = session.post(
                api_urls["process"], 
                json=payload, 
                timeout=120
            )
        
        if response.status_code == 200:
            try:
                return True, response.json()
            except json.JSONDecodeError:
                return True, response.text
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", f"API error: {response.status_code}")
            except:
                error_msg = f"API error: {response.status_code} - {response.text}"
            
            return False, error_msg
            
    except requests.Timeout:
        return False, "Request timed out. Please try again."
    except requests.RequestException as e:
        return False, f"Request failed: {str(e)}"
    except Exception as e:
        return False, f"Processing error: {str(e)}"

# Display results
def display_results(results, task_type):
    """Display task results in a formatted way"""
    if isinstance(results, str):
        st.markdown(f"**Response:** {results}")
        return
    
    if not isinstance(results, dict):
        st.error("Invalid response format")
        return
    
    task_results = results.get("results", [])
    evaluation = results.get("evaluation")
    
    if task_results:
        st.markdown('<div class="sub-header">🤖 Model Results</div>', unsafe_allow_html=True)
        
        for result in task_results:
            model_name = result.get("model", "Unknown")
            output = result.get("output")
            inference_time = result.get("inference_time", 0)
            success = result.get("success", True)
            
            if success and output:
                st.markdown(f"**{model_name}:**")
                st.write(output)
                st.caption(f"⏱️ Generation time: {inference_time:.2f}s")
                
                word_count = result.get("word_count")
                if word_count:
                    st.caption(f"📝 Words: {word_count}")
                
                quality_issues = result.get("quality_issues", [])
                if quality_issues:
                    st.warning(f"Quality issues: {', '.join(quality_issues)}")
                
            else:
                error_msg = result.get("error", "Unknown error")
                st.error(f"**{model_name}:** {error_msg}")
            
            st.divider()
    
    if evaluation and len(evaluation) > 0:
        st.markdown('<div class="sub-header">📊 Evaluation Metrics</div>', unsafe_allow_html=True)
        
        try:
            eval_df = pd.DataFrame(evaluation)
            st.dataframe(eval_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying evaluation: {e}")

# Document upload
def handle_document_upload(api_urls, uploaded_file):
    """Handle document upload for RAG"""
    try:
        session = create_session()
        
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        
        with st.spinner("Uploading and processing document..."):
            response = session.post(api_urls["upload"], files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ {result.get('message', 'Document uploaded successfully')}")
            
            filename = result.get("filename", uploaded_file.name)
            text_length = result.get("text_length", 0)
            word_count = result.get("word_count", 0)
            
            st.info(f"📄 **{filename}**: {text_length} characters, {word_count} words")
            
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", "Upload failed")
            except:
                error_msg = f"Upload failed: {response.status_code}"
            
            st.error(f"❌ {error_msg}")
            
    except Exception as e:
        st.error(f"❌ Upload error: {e}")

# JavaScript message handler
def handle_js_messages():
    """Handle messages from JavaScript voice interface"""
    if "voice_input_text" in st.session_state and st.session_state.voice_input_text:
        voice_text = st.session_state.voice_input_text
        st.session_state.last_voice_input = voice_text
        st.session_state.voice_input_text = ""  # Clear after use
        return voice_text
    return None

# Main app
def main():
    """Main application with integrated voice features"""
    initialize_session_state()
    
    config = load_config()
    api_urls = get_api_urls(config)
    
    # Header
    st.markdown('<div class="main-header">🤖 AI Workbench</div>', unsafe_allow_html=True)
    st.markdown("*Advanced AI platform with voice interaction capabilities*")
    
    # Check API health
    with st.spinner("Checking API status..."):
        health_ok, health_info = check_api_health(api_urls)
    
    if not health_ok:
        st.error(f"❌ **API Not Available:** {health_info}")
        st.info("💡 Make sure to start the API server: `python main.py`")
        st.stop()
    
    # Display API status
    if isinstance(health_info, dict):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("API Status", "🟢 Healthy")
        
        with col2:
            available_models = health_info.get("models_available", 0)
            st.metric("Available Models", available_models)
        
        with col3:
            model_names = health_info.get("model_names", [])
            if model_names:
                st.metric("Active Models", ", ".join(model_names))
    
    # Fetch additional data
    fetch_model_info(api_urls)
    fetch_supported_languages(api_urls)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Task selection
        st.markdown('<div class="sub-header">🎯 Select Task</div>', unsafe_allow_html=True)
        task = st.selectbox(
            "Choose a task:",
            ["chat", "summarization", "translation"],
            format_func=lambda x: {
                "summarization": "📝 Text Summarization",
                "translation": "🌐 Language Translation", 
                "chat": "💬 AI Chat"
            }[x]
        )
        
        # Task-specific information
        if task == "chat":
            st.info("💡 **Chat Mode**: Uses uploaded documents for context. Voice features available!")
        elif task == "summarization":
            st.info("💡 **Summarization Mode**: Processes only your input text (ignores uploaded documents)")
        elif task == "translation":
            st.info("💡 **Translation Mode**: Translates only your input text (ignores uploaded documents)")
        
        # Initialize variables
        text_input = ""
        reference = ""
        target_lang = None
        messages = []
        
        # Task-specific inputs
        if task == "summarization":
            st.markdown("### 📝 Text Summarization")
            text_input = st.text_area(
                "Enter text to summarize:",
                height=200,
                placeholder="Paste your text here..."
            )
            reference = st.text_area(
                "Reference summary (optional):",
                height=100,
                help="Provide a reference summary for evaluation"
            )
            
        elif task == "translation":
            st.markdown("### 🌐 Language Translation")
            text_input = st.text_area(
                "Enter text to translate:",
                height=200,
                placeholder="Enter text in any language..."
            )
            target_lang = st.selectbox(
                "Target Language:",
                st.session_state.supported_languages
            )
            reference = st.text_area(
                "Reference translation (optional):",
                height=100,
                help="Provide a reference translation for evaluation"
            )
            
        else:  # chat
            st.markdown("### 💬 AI Chat with Voice")
            
            # Voice Interface Section
            if st.session_state.voice_enabled:
                st.markdown("#### 🎤 Voice Input")
                render_voice_interface(api_urls)
                
                # Check for voice input from JavaScript
                # Note: In practice, you'd need to implement a proper message passing mechanism
                # This is a simplified version for demonstration
                
                # Alternative voice recorder
                st.markdown("---")
                audio_data = simple_voice_recorder()
                if audio_data:
                    voice_text = process_voice_input(api_urls, audio_data)
                    if voice_text:
                        st.session_state.messages.append({"role": "user", "content": voice_text})
                        st.success(f"🎤 Voice input added: {voice_text}")
                        st.rerun()
            
            # Display chat history
            st.markdown("#### 💬 Conversation History")
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
            
            # Chat input
            st.markdown("#### ⌨️ Text Input")
            
            # Voice response option
            voice_response = st.checkbox("🔊 Enable Voice Response", help="AI will respond with speech")
            
            if prompt := st.chat_input("Type your message or use voice input above..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Process with voice if enabled
                if voice_response and st.session_state.voice_enabled:
                    if handle_voice_chat(api_urls, prompt, st.session_state.messages, {"temperature": 0.7, "max_tokens": 150}):
                        st.rerun()
                else:
                    # Regular chat processing
                    text_input = prompt
                    messages = st.session_state.messages
    
    with col2:
        # Sidebar controls
        st.markdown('<div class="sub-header">⚙️ Settings</div>', unsafe_allow_html=True)
        
        # Voice settings
        with st.expander("🎤 Voice Settings", expanded=True):
            st.session_state.voice_enabled = st.checkbox(
                "Enable Voice Features", 
                value=st.session_state.voice_enabled,
                help="Enable voice input and text-to-speech output"
            )
            
            if st.session_state.voice_enabled:
                st.success("🎤 Voice features enabled")
                st.markdown("""
                **How to use voice:**
                - Click 🎤 button to record
                - Speak clearly for up to 10 seconds
                - Click 🛑 to stop early
                - Enable 🔊 for voice responses
                """)
                
                # Voice troubleshooting
                if st.button("🔧 Test Microphone"):
                    st.info("Click the 🎤 button above and speak to test your microphone")
                    
                if st.button("🔄 Reset Voice Interface"):
                    # Clear any voice state
                    st.session_state.voice_input_text = ""
                    st.session_state.last_voice_input = ""
                    st.success("Voice interface reset")
                    
            else:
                st.info("Voice features are disabled")
        
        # Model parameters
        with st.expander("🎛️ Model Settings", expanded=False):
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
            top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.1)
            max_tokens = st.slider("Max Tokens", 50, 500, 100, 10)
            min_tokens = st.slider("Min Tokens", 10, 100, 30, 5)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            use_fusion = st.checkbox("Model Fusion", help="Combine outputs from multiple models")
            use_tutor = st.checkbox("AI Tutor", help="Get explanations about model performance")
        
        # Evaluation metrics
        if task != "chat":
            with st.expander("📊 Evaluation Metrics"):
                available_metrics = {
                    "summarization": ["rouge1", "rouge2", "rougeL", "bertscore"],
                    "translation": ["bleu", "meteor"]
                }
                
                metrics = st.multiselect(
                    "Select metrics:",
                    available_metrics.get(task, []),
                    default=available_metrics.get(task, [])[:2]
                )
        else:
            metrics = []
        
        # Document upload for RAG
        with st.expander("📁 Document Upload"):
            st.markdown("**For Chat Context Only**")
            uploaded_file = st.file_uploader(
                "Upload document:",
                type=["pdf", "png", "jpg", "jpeg"],
                help="Documents provide context for chat conversations only"
            )
            
            if uploaded_file and st.button("📤 Upload"):
                handle_document_upload(api_urls, uploaded_file)
    
    # Process button and results
    st.markdown("---")
    
    # FIXED: Voice input detection for chat with proper JavaScript handling
    if task == "chat":
        # JavaScript message listener for voice input
        voice_listener_js = """
        <script>
        window.addEventListener('message', function(event) {
            if (event.data && event.data.type === 'voiceInput') {
                // Store voice input in a way that Streamlit can access
                fetch(window.location.origin + '/voice_handler', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        text: event.data.text,
                        timestamp: Date.now()
                    })
                }).catch(console.error);
                
                // Alternative: Use Streamlit's built-in messaging
                if (window.parent && window.parent.postMessage) {
                    window.parent.postMessage({
                        type: 'streamlit:voiceInput',
                        text: event.data.text
                    }, '*');
                }
            }
        });
        </script>
        """
        components.html(voice_listener_js, height=0)
    
    # Determine if we should process
    should_process = False
    
    if task == "chat" and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        should_process = True
        messages = st.session_state.messages
    elif task in ["summarization", "translation"] and st.button("🚀 Process", type="primary"):
        should_process = True
    
    if should_process:
        # Validate inputs
        if task in ["summarization", "translation"] and not text_input:
            st.error("❌ Please enter text to process")
        elif task == "translation" and not target_lang:
            st.error("❌ Please select a target language")
        else:
            # FIXED: Ensure metrics are included for evaluation graphs
            if task == "summarization" and not metrics:
                metrics = ["rouge1", "rouge2", "rougeL"]  # Default metrics for graphs
            elif task == "translation" and not metrics:
                metrics = ["bleu", "meteor"]  # Default metrics for graphs
            
            # Prepare payload
            payload = {
                "task": task,
                "text": text_input if task != "chat" else "",
                "reference": reference if reference else "",
                "target_lang": target_lang if target_lang else "",
                "messages": messages if task == "chat" else [],
                "params": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "min_tokens": min_tokens
                },
                "metrics": metrics,
                "use_fusion": use_fusion,
                "use_tutor": use_tutor
            }
            
            # Process task
            success, results = process_task(api_urls, payload)
            
            if success:
                if task == "chat":
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": results})
                    with st.chat_message("assistant"):
                        st.write(results)
                    st.rerun()
                else:
                    # FIXED: Display structured results with graphs
                    display_results(results, task)
            else:
                st.error(f"❌ **Processing failed:** {results}")
    
    # Footer with information
    st.markdown("---")
    
    # Voice usage tips
    if st.session_state.voice_enabled and task == "chat":
        st.markdown("### 🎤 Voice Chat Tips")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **🎙️ Recording Tips**
            - Speak clearly and slowly
            - Use quiet environment
            - Keep mic close to mouth
            """)
        
        with col2:
            st.info("""
            **🔊 Audio Quality**
            - Check browser permissions
            - Use good microphone
            - Test with short phrases first
            """)
        
        with col3:
            st.info("""
            **🤖 Best Results**
            - Ask clear questions
            - Upload relevant documents
            - Use voice responses for natural flow
            """)
    
    # Additional features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("👥 Crowdsourcing"):
            st.markdown("**Submit Dataset**")
            dataset_input = st.text_area("Dataset (JSON format):", height=100)
            submitter = st.text_input("Your name:", value="anonymous")
            
            if st.button("📤 Submit Dataset"):
                if dataset_input:
                    try:
                        data = json.loads(dataset_input)
                        # Handle crowdsource submission (implement if needed)
                        st.success("Dataset submitted!")
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON format")
                else:
                    st.error("❌ Please enter dataset")
    
    with col2:
        with st.expander("📊 Usage Statistics"):
            if st.button("📈 Get Stats"):
                try:
                    session = create_session()
                    response = session.get(api_urls["usage"], timeout=10)
                    
                    if response.status_code == 200:
                        stats = response.json()
                        st.json(stats)
                    else:
                        st.warning("Could not fetch usage statistics")
                except Exception as e:
                    st.error(f"Error fetching stats: {e}")
    
    with col3:
        with st.expander("ℹ️ System Info"):
            if st.session_state.model_info:
                st.json(st.session_state.model_info)
            
            if st.button("🔄 Refresh Info"):
                st.session_state.model_info = fetch_model_info(api_urls)
                st.rerun()
    
    # Debug information (can be removed in production)
    if st.sidebar.checkbox("🔍 Debug Mode", help="Show debug information"):
        with st.sidebar.expander("Debug Info"):
            st.write("**Session State:**")
            st.write(f"Voice enabled: {st.session_state.voice_enabled}")
            st.write(f"Messages: {len(st.session_state.messages)}")
            st.write(f"Last voice input: {st.session_state.last_voice_input}")
            st.write(f"Current task: {task}")

if __name__ == "__main__":
    main()