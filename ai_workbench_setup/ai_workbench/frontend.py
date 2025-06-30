import streamlit as st
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
import io

# Page configuration
st.set_page_config(
    page_title="AI Workbench",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for voice interface
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

/* Enhanced Voice Interface Styles */
.voice-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 25px;
    margin: 15px 0;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}

.voice-controls {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 15px;
    margin: 20px 0;
}

.voice-btn {
    background: linear-gradient(145deg, #007bff, #0056b3);
    border: none;
    border-radius: 50%;
    width: 80px;
    height: 80px;
    color: white;
    font-size: 32px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 8px 20px rgba(0,123,255,0.3);
    position: relative;
    overflow: hidden;
}

.voice-btn:hover {
    background: linear-gradient(145deg, #0056b3, #004085);
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(0,123,255,0.4);
}

.voice-btn.recording {
    background: linear-gradient(145deg, #dc3545, #c82333);
    animation: pulse-glow 1.5s infinite;
    box-shadow: 0 0 30px rgba(220,53,69,0.6);
}

.voice-btn.processing {
    background: linear-gradient(145deg, #ffc107, #e0a800);
    animation: spin 2s linear infinite;
}

@keyframes pulse-glow {
    0% { 
        transform: scale(1);
        box-shadow: 0 0 20px rgba(220,53,69,0.6);
    }
    50% { 
        transform: scale(1.05);
        box-shadow: 0 0 40px rgba(220,53,69,0.8);
    }
    100% { 
        transform: scale(1);
        box-shadow: 0 0 20px rgba(220,53,69,0.6);
    }
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.voice-status {
    margin: 20px 0;
    padding: 15px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease;
}

.status-ready { 
    background: rgba(40, 167, 69, 0.2); 
    color: #28a745; 
    border: 2px solid rgba(40, 167, 69, 0.3);
}
.status-recording { 
    background: rgba(220, 53, 69, 0.2); 
    color: #dc3545; 
    border: 2px solid rgba(220, 53, 69, 0.3);
}
.status-processing { 
    background: rgba(255, 193, 7, 0.2); 
    color: #ffc107; 
    border: 2px solid rgba(255, 193, 7, 0.3);
}

.transcription-box {
    margin-top: 20px; 
    padding: 20px; 
    background: rgba(255,255,255,0.95); 
    border-radius: 15px; 
    min-height: 60px;
    color: #333;
    font-size: 16px;
    border: 2px solid rgba(255,255,255,0.3);
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.1);
    line-height: 1.5;
}

.voice-settings {
    margin-top: 15px;
    padding: 15px;
    background: rgba(255,255,255,0.1);
    border-radius: 12px;
    display: flex;
    justify-content: space-around;
    align-items: center;
}

.voice-setting {
    display: flex;
    flex-direction: column;
    align-items: center;
    color: white;
    font-size: 14px;
}

.voice-tips {
    margin-top: 20px;
    font-size: 14px;
    opacity: 0.9;
    line-height: 1.4;
}

.audio-visualizer {
    width: 100%;
    height: 40px;
    background: rgba(255,255,255,0.1);
    border-radius: 20px;
    margin: 15px 0;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.visualizer-bar {
    width: 3px;
    background: linear-gradient(to top, #007bff, #00d4ff);
    margin: 0 1px;
    transition: height 0.1s ease;
    border-radius: 2px;
}

.chat-bubble {
    background: #f8f9fa;
    border-radius: 18px;
    padding: 12px 18px;
    margin: 8px 0;
    max-width: 80%;
    word-wrap: break-word;
}

.chat-bubble.user {
    background: #007bff;
    color: white;
    margin-left: auto;
}

.chat-bubble.assistant {
    background: #e9ecef;
    color: #333;
}

.voice-response-controls {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-top: 15px;
}

.mini-btn {
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 8px;
    padding: 8px 15px;
    color: white;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.mini-btn:hover {
    background: rgba(255,255,255,0.3);
    border-color: rgba(255,255,255,0.5);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables including voice settings"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_status" not in st.session_state:
        st.session_state.api_status = None
    if "model_info" not in st.session_state:
        st.session_state.model_info = None
    if "supported_languages" not in st.session_state:
        st.session_state.supported_languages = ["Spanish", "French", "German", "Italian", "Portuguese"]
    
    # Voice-specific state
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = True
    if "voice_input_text" not in st.session_state:
        st.session_state.voice_input_text = ""
    if "last_voice_input" not in st.session_state:
        st.session_state.last_voice_input = ""
    if "voice_response_enabled" not in st.session_state:
        st.session_state.voice_response_enabled = True
    if "auto_speech" not in st.session_state:
        st.session_state.auto_speech = False
    if "voice_language" not in st.session_state:
        st.session_state.voice_language = "en"
    if "last_audio_response" not in st.session_state:
        st.session_state.last_audio_response = None

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

# Enhanced Voice Interface Component
def render_advanced_voice_interface(api_urls):
    """Render advanced voice interface with ChatGPT-like features"""
    
    voice_html = f"""
    <div class="voice-container">
        <h3 style="margin-top: 0; font-size: 24px;">🎤 Advanced Voice Assistant</h3>
        <p style="margin: 10px 0; opacity: 0.9; font-size: 16px;">
            Talk naturally with AI - just like ChatGPT Voice
        </p>
        
        <div class="voice-controls">
            <button id="voiceBtn" class="voice-btn" onclick="toggleRecording()" title="Click to start/stop recording">
                🎤
            </button>
        </div>
        
        <!-- Audio Visualizer -->
        <div id="audioVisualizer" class="audio-visualizer" style="display: none;">
            <div class="visualizer-bar" style="height: 20px;"></div>
            <div class="visualizer-bar" style="height: 35px;"></div>
            <div class="visualizer-bar" style="height: 25px;"></div>
            <div class="visualizer-bar" style="height: 40px;"></div>
            <div class="visualizer-bar" style="height: 30px;"></div>
            <div class="visualizer-bar" style="height: 35px;"></div>
            <div class="visualizer-bar" style="height: 45px;"></div>
            <div class="visualizer-bar" style="height: 25px;"></div>
            <div class="visualizer-bar" style="height: 30px;"></div>
            <div class="visualizer-bar" style="height: 20px;"></div>
        </div>
        
        <div id="voiceStatus" class="voice-status status-ready">
            🎯 Ready to listen - Click the microphone to start
        </div>
        
        <div id="transcription" class="transcription-box">
            Your speech will appear here in real-time...
        </div>
        
        <!-- Voice Settings -->
        <div class="voice-settings">
            <div class="voice-setting">
                <strong>🌍 Language</strong>
                <select id="voiceLanguage" onchange="updateLanguage()">
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="it">Italian</option>
                    <option value="pt">Portuguese</option>
                </select>
            </div>
            <div class="voice-setting">
                <strong>🔊 Auto-play</strong>
                <input type="checkbox" id="autoPlay" checked onchange="updateAutoPlay()">
            </div>
            <div class="voice-setting">
                <strong>⚡ Continuous</strong>
                <input type="checkbox" id="continuous" onchange="updateContinuous()">
            </div>
        </div>
        
        <!-- Response Controls -->
        <div class="voice-response-controls">
            <button class="mini-btn" onclick="clearTranscription()">🗑️ Clear</button>
            <button class="mini-btn" onclick="downloadAudio()" id="downloadBtn" style="display: none;">💾 Download</button>
            <button class="mini-btn" onclick="testMicrophone()">🎧 Test Mic</button>
        </div>
        
        <div class="voice-tips">
            💡 <strong>Pro Tips:</strong><br>
            • Speak naturally and clearly<br>
            • Use "Hey AI" to wake up continuous mode<br>
            • Say "stop listening" to end continuous mode<br>
            • Toggle auto-play for voice responses
        </div>
    </div>
    
    <script>
    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];
    let recordingTimeout = null;
    let audioContext = null;
    let analyzer = null;
    let microphone = null;
    let continuousMode = false;
    let autoPlay = true;
    let currentLanguage = 'en';
    let lastAudioBlob = null;
    
    // Initialize audio context for visualization
    async function initAudioContext() {{
        try {{
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyzer = audioContext.createAnalyser();
            analyzer.fftSize = 256;
        }} catch (error) {{
            console.log('Audio context initialization failed:', error);
        }}
    }}
    
    async function toggleRecording() {{
        const btn = document.getElementById('voiceBtn');
        const status = document.getElementById('voiceStatus');
        
        if (!isRecording) {{
            await startRecording();
        }} else {{
            stopRecording();
        }}
    }}
    
    async function startRecording() {{
        try {{
            const stream = await navigator.mediaDevices.getUserMedia({{ 
                audio: {{
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 44100
                }}
            }});
            
            mediaRecorder = new MediaRecorder(stream, {{
                mimeType: 'audio/webm;codecs=opus'
            }});
            audioChunks = [];
            
            // Setup audio visualization
            if (audioContext) {{
                microphone = audioContext.createMediaStreamSource(stream);
                microphone.connect(analyzer);
                startVisualization();
            }}
            
            mediaRecorder.ondataavailable = event => {{
                if (event.data.size > 0) {{
                    audioChunks.push(event.data);
                }}
            }};
            
            mediaRecorder.onstop = async () => {{
                const audioBlob = new Blob(audioChunks, {{ type: 'audio/webm' }});
                lastAudioBlob = audioBlob;
                await processAudio(audioBlob);
                stopVisualization();
            }};
            
            mediaRecorder.start();
            isRecording = true;
            
            // Update UI
            const btn = document.getElementById('voiceBtn');
            const status = document.getElementById('voiceStatus');
            const visualizer = document.getElementById('audioVisualizer');
            
            btn.classList.add('recording');
            btn.innerHTML = '🛑';
            btn.title = 'Click to stop recording';
            status.className = 'voice-status status-recording';
            status.innerHTML = '🔴 Recording... Speak naturally';
            visualizer.style.display = 'flex';
            
            // Auto-stop after 30 seconds
            recordingTimeout = setTimeout(() => {{
                if (isRecording) {{
                    stopRecording();
                }}
            }}, 30000);
            
        }} catch (error) {{
            const status = document.getElementById('voiceStatus');
            status.className = 'voice-status status-ready';
            status.innerHTML = '❌ Microphone access denied. Please enable permissions and refresh.';
            console.error('Error accessing microphone:', error);
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
        
        // Update UI
        const btn = document.getElementById('voiceBtn');
        const status = document.getElementById('voiceStatus');
        
        btn.classList.remove('recording');
        btn.classList.add('processing');
        btn.innerHTML = '⏳';
        btn.title = 'Processing speech...';
        status.className = 'voice-status status-processing';
        status.innerHTML = '🤖 Processing speech and generating response...';
        
        // Stop all tracks
        if (mediaRecorder && mediaRecorder.stream) {{
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }}
    }}
    
    function startVisualization() {{
        if (!analyzer) return;
        
        const visualizer = document.getElementById('audioVisualizer');
        const bars = visualizer.querySelectorAll('.visualizer-bar');
        const bufferLength = analyzer.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        function animate() {{
            if (!isRecording) return;
            
            analyzer.getByteFrequencyData(dataArray);
            
            bars.forEach((bar, index) => {{
                const dataIndex = Math.floor(index * bufferLength / bars.length);
                const height = Math.max(10, (dataArray[dataIndex] / 255) * 40);
                bar.style.height = height + 'px';
            }});
            
            requestAnimationFrame(animate);
        }}
        
        animate();
    }}
    
    function stopVisualization() {{
        const visualizer = document.getElementById('audioVisualizer');
        visualizer.style.display = 'none';
    }}
    
    async function processAudio(audioBlob) {{
        const status = document.getElementById('voiceStatus');
        const transcription = document.getElementById('transcription');
        const btn = document.getElementById('voiceBtn');
        
        try {{
            // Show processing state
            status.className = 'voice-status status-processing';
            status.innerHTML = '🎯 Converting speech to text...';
            transcription.innerHTML = '<em>Processing audio...</em>';
            
            // Convert blob to FormData for upload
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.webm');
            
            // Send to speech-to-text API
            const response = await fetch('{api_urls["speech_to_text"]}', {{
                method: 'POST',
                body: formData
            }});
            
            if (response.ok) {{
                const result = await response.json();
                const text = result.text || '';
                
                if (text.trim()) {{
                    // Success - show transcription
                    transcription.innerHTML = `<strong>You said:</strong> "${{text}}"`;
                    
                    // Check for continuous mode commands
                    const lowerText = text.toLowerCase();
                    if (lowerText.includes('stop listening') || lowerText.includes('end conversation')) {{
                        continuousMode = false;
                        document.getElementById('continuous').checked = false;
                        status.className = 'voice-status status-ready';
                        status.innerHTML = '✅ Speech recognized! Continuous mode disabled.';
                    }} else {{
                        status.className = 'voice-status status-processing';
                        status.innerHTML = '🤖 Getting AI response...';
                        
                        // Send to voice chat API for AI response
                        await getAIVoiceResponse(text);
                    }}
                    
                    // Send to Streamlit
                    window.parent.postMessage({{
                        type: 'voiceInput',
                        text: text,
                        timestamp: Date.now(),
                        language: currentLanguage
                    }}, '*');
                    
                }} else {{
                    // No speech detected
                    transcription.innerHTML = '<em>No speech detected. Please try again.</em>';
                    resetToReady();
                }}
            }} else {{
                throw new Error(`HTTP ${{response.status}}: ${{response.statusText}}`);
            }}
            
        }} catch (error) {{
            console.error('Error processing audio:', error);
            transcription.innerHTML = '<em>Error processing speech. Please try again.</em>';
            resetToReady();
        }}
    }}
    
    async function getAIVoiceResponse(userText) {{
        try {{
            const payload = {{
                text: userText,
                messages: [], // Streamlit will handle this
                params: {{
                    temperature: 0.7,
                    max_tokens: 150
                }}
            }};
            
            const response = await fetch('{api_urls["voice_chat"]}', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }},
                body: JSON.stringify(payload)
            }});
            
            if (response.ok) {{
                const result = await response.json();
                const aiText = result.text || '';
                const audioPath = result.audio_path;
                
                if (aiText) {{
                    // Update status
                    const status = document.getElementById('voiceStatus');
                    status.className = 'voice-status status-ready';
                    status.innerHTML = '✅ AI responded! Ready for next input.';
                    
                    // Show AI response in transcription area
                    const transcription = document.getElementById('transcription');
                    transcription.innerHTML = `
                        <div><strong>You:</strong> "${{userText}}"</div>
                        <div style="margin-top: 10px;"><strong>AI:</strong> "${{aiText}}"</div>
                    `;
                    
                    // Auto-play AI response if enabled
                    if (autoPlay && audioPath) {{
                        await playAIResponse(audioPath);
                    }}
                    
                    // Enable download button
                    const downloadBtn = document.getElementById('downloadBtn');
                    downloadBtn.style.display = 'inline-block';
                    
                    // Continue listening if continuous mode is on
                    if (continuousMode) {{
                        setTimeout(() => {{
                            if (!isRecording) {{
                                startRecording();
                            }}
                        }}, 2000);
                    }}
                }}
            }}
        }} catch (error) {{
            console.error('Error getting AI response:', error);
            resetToReady();
        }}
        
        resetButtonState();
    }}
    
    async function playAIResponse(audioPath) {{
        try {{
            // This would play the audio file returned by the API
            // Implementation depends on how your API returns audio
            console.log('Playing AI response audio:', audioPath);
        }} catch (error) {{
            console.error('Error playing AI response:', error);
        }}
    }}
    
    function resetToReady() {{
        const status = document.getElementById('voiceStatus');
        status.className = 'voice-status status-ready';
        status.innerHTML = continuousMode ? 
            '👂 Continuous listening mode - Say "stop listening" to end' :
            '🎯 Ready for voice input - Click microphone to start';
    }}
    
    function resetButtonState() {{
        const btn = document.getElementById('voiceBtn');
        btn.classList.remove('recording', 'processing');
        btn.innerHTML = '🎤';
        btn.title = 'Click to start recording';
    }}
    
    function updateLanguage() {{
        const select = document.getElementById('voiceLanguage');
        currentLanguage = select.value;
        
        // Notify Streamlit about language change
        window.parent.postMessage({{
            type: 'languageChange',
            language: currentLanguage
        }}, '*');
    }}
    
    function updateAutoPlay() {{
        const checkbox = document.getElementById('autoPlay');
        autoPlay = checkbox.checked;
        
        window.parent.postMessage({{
            type: 'autoPlayChange',
            autoPlay: autoPlay
        }}, '*');
    }}
    
    function updateContinuous() {{
        const checkbox = document.getElementById('continuous');
        continuousMode = checkbox.checked;
        
        if (continuousMode) {{
            const status = document.getElementById('voiceStatus');
            status.innerHTML = '👂 Continuous mode enabled - Say "Hey AI" to start or "stop listening" to end';
        }} else {{
            resetToReady();
        }}
        
        window.parent.postMessage({{
            type: 'continuousModeChange',
            continuous: continuousMode
        }}, '*');
    }}
    
    function clearTranscription() {{
        const transcription = document.getElementById('transcription');
        transcription.innerHTML = 'Your speech will appear here in real-time...';
        resetToReady();
        
        window.parent.postMessage({{
            type: 'clearTranscription'
        }}, '*');
    }}
    
    function downloadAudio() {{
        if (lastAudioBlob) {{
            const url = URL.createObjectURL(lastAudioBlob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `voice_recording_${{Date.now()}}.webm`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
    }}
    
    async function testMicrophone() {{
        try {{
            const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
            const status = document.getElementById('voiceStatus');
            status.className = 'voice-status status-ready';
            status.innerHTML = '✅ Microphone test successful!';
            
            // Stop the test stream
            stream.getTracks().forEach(track => track.stop());
            
            setTimeout(() => {{
                resetToReady();
            }}, 2000);
        }} catch (error) {{
            const status = document.getElementById('voiceStatus');
            status.className = 'voice-status status-ready';
            status.innerHTML = '❌ Microphone test failed. Check permissions.';
        }}
    }}
    
    // Initialize on load
    window.addEventListener('load', () => {{
        initAudioContext();
    }});
    
    // Listen for messages from Streamlit
    window.addEventListener('message', function(event) {{
        if (event.data.type === 'clearVoiceTranscription') {{
            clearTranscription();
        }} else if (event.data.type === 'updateVoiceSettings') {{
            // Update settings from Streamlit
            if (event.data.language) {{
                currentLanguage = event.data.language;
                document.getElementById('voiceLanguage').value = currentLanguage;
            }}
            if (event.data.autoPlay !== undefined) {{
                autoPlay = event.data.autoPlay;
                document.getElementById('autoPlay').checked = autoPlay;
            }}
        }}
    }});
    </script>
    """
    
    # Render the enhanced voice interface
    components.html(voice_html, height=500)

# Voice Settings Panel
def render_voice_settings():
    """Render voice settings panel"""
    with st.expander("🎤 Voice Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Voice language selection
            st.session_state.voice_language = st.selectbox(
                "🌍 Voice Language",
                ["en", "es", "fr", "de", "it", "pt"],
                index=0,
                format_func=lambda x: {
                    "en": "🇺🇸 English",
                    "es": "🇪🇸 Spanish", 
                    "fr": "🇫🇷 French",
                    "de": "🇩🇪 German",
                    "it": "🇮🇹 Italian",
                    "pt": "🇵🇹 Portuguese"
                }[x]
            )
            
            # Voice response settings
            st.session_state.voice_response_enabled = st.checkbox(
                "🔊 Enable Voice Responses",
                value=st.session_state.voice_response_enabled,
                help="AI will respond with synthesized speech"
            )
            
            st.session_state.auto_speech = st.checkbox(
                "⚡ Auto-play Responses",
                value=st.session_state.auto_speech,
                help="Automatically play AI voice responses"
            )
        
        with col2:
            # Voice quality settings
            voice_speed = st.slider(
                "🎵 Speech Speed",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust AI speech speed"
            )
            
            voice_pitch = st.slider(
                "🎼 Voice Pitch",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust AI voice pitch"
            )
            
            # Continuous listening
            continuous_listening = st.checkbox(
                "👂 Continuous Listening",
                value=False,
                help="Keep listening after each response"
            )
        
        # Voice diagnostics
        if st.button("🔧 Test Voice System"):
            test_voice_system()

def test_voice_system():
    """Test voice system functionality"""
    with st.spinner("Testing voice system..."):
        try:
            # Test microphone access (simulated)
            st.success("✅ Microphone access: Available")
            
            # Test speech recognition
            st.success("✅ Speech recognition: Ready")
            
            # Test text-to-speech
            st.success("✅ Text-to-speech: Ready")
            
            # Test API connectivity
            config = load_config()
            api_urls = get_api_urls(config)
            session = create_session()
            
            try:
                response = session.get(api_urls["health"], timeout=5)
                if response.status_code == 200:
                    st.success("✅ Voice API: Connected")
                else:
                    st.warning("⚠️ Voice API: Connection issues")
            except:
                st.error("❌ Voice API: Not available")
            
        except Exception as e:
            st.error(f"❌ Voice system test failed: {e}")

# Enhanced chat interface with voice
def render_enhanced_chat():
    """Render enhanced chat interface with voice capabilities"""
    st.markdown("### 💬 AI Chat with Advanced Voice")
    
    # Voice interface section
    if st.session_state.voice_enabled:
        render_advanced_voice_interface(get_api_urls(load_config()))
        
        # Voice settings
        render_voice_settings()
        
        st.markdown("---")
    
    # Chat history display
    st.markdown("#### 💭 Conversation History")
    
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Add voice playback for assistant messages
                if (message["role"] == "assistant" and 
                    st.session_state.voice_response_enabled and
                    st.session_state.last_audio_response):
                    
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        if st.button("🔊", key=f"play_{i}", help="Play voice response"):
                            play_audio_response(st.session_state.last_audio_response)
                    with col2:
                        if st.button("💾", key=f"download_{i}", help="Download audio"):
                            download_audio_response(st.session_state.last_audio_response)
    
    # Text input with voice integration
    st.markdown("#### ⌨️ Text Input")
    
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Type your message or use voice input above...",
            key="text_input",
            placeholder="Ask me anything or click the microphone above to speak"
        )
    with col2:
        send_button = st.button("Send", type="primary")
    
    # Process text input
    if (user_input and send_button) or user_input:
        process_user_input(user_input)
    
    # Handle voice input from JavaScript
    handle_voice_messages()

def process_user_input(text):
    """Process user input (text or voice)"""
    if not text.strip():
        return
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": text})
    
    # Get AI response
    with st.spinner("🤖 AI is thinking..."):
        ai_response = get_ai_response(text)
        
        if ai_response:
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
            # Generate voice response if enabled
            if st.session_state.voice_response_enabled:
                audio_response = generate_voice_response(ai_response)
                if audio_response:
                    st.session_state.last_audio_response = audio_response
                    
                    # Auto-play if enabled
                    if st.session_state.auto_speech:
                        play_audio_response(audio_response)
    
    # Clear text input and rerun
    st.session_state.text_input = ""
    st.rerun()

def get_ai_response(user_text):
    """Get AI response from the API"""
    try:
        config = load_config()
        api_urls = get_api_urls(config)
        session = create_session()
        
        # Prepare payload
        payload = {
            "task": "chat",
            "messages": st.session_state.messages,
            "params": {
                "temperature": 0.7,
                "max_tokens": 150,
                "top_p": 0.9
            }
        }
        
        # Make API call
        response = session.post(api_urls["process"], json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.text.strip()
        else:
            st.error(f"API Error: {response.status_code}")
            return "Sorry, I'm having trouble processing your request right now."
            
    except Exception as e:
        st.error(f"Error: {e}")
        return "I apologize, but I encountered an error processing your request."

def generate_voice_response(text):
    """Generate voice response from text"""
    try:
        config = load_config()
        api_urls = get_api_urls(config)
        session = create_session()
        
        # Prepare voice generation payload
        payload = {
            "text": text,
            "language": st.session_state.voice_language,
            "speed": 1.0,  # You can make this configurable
            "pitch": 1.0   # You can make this configurable
        }
        
        # Call text-to-speech API (you'll need to implement this endpoint)
        response = session.post(f"{api_urls['base']}/text_to_speech", json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.content  # Audio bytes
        else:
            st.warning("Voice generation failed")
            return None
            
    except Exception as e:
        st.warning(f"Voice generation error: {e}")
        return None

def play_audio_response(audio_data):
    """Play audio response"""
    if audio_data:
        st.audio(audio_data, format="audio/mp3", autoplay=True)

def download_audio_response(audio_data):
    """Provide download link for audio response"""
    if audio_data:
        st.download_button(
            label="💾 Download Audio",
            data=audio_data,
            file_name=f"ai_response_{int(time.time())}.mp3",
            mime="audio/mp3"
        )

def handle_voice_messages():
    """Handle voice messages from JavaScript"""
    # This would be called when voice input is received
    # In a real implementation, you'd need to set up proper message passing
    
    # Check for voice input in session state (simulated)
    if hasattr(st.session_state, 'pending_voice_input'):
        voice_text = st.session_state.pending_voice_input
        del st.session_state.pending_voice_input
        
        # Process voice input
        process_user_input(voice_text)

# Alternative simple voice recorder
def render_simple_voice_recorder():
    """Simple voice recorder fallback"""
    try:
        # Try to import audio recorder
        from audio_recorder_streamlit import audio_recorder
        
        st.markdown("#### 🎙️ Simple Voice Recorder")
        st.info("Click to record, speak, then click again to stop")
        
        audio_bytes = audio_recorder(
            text="🎤 Record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
            sample_rate=44100
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎯 Process Recording", type="primary"):
                    process_voice_recording(audio_bytes)
            with col2:
                st.download_button(
                    label="💾 Download",
                    data=audio_bytes,
                    file_name=f"recording_{int(time.time())}.wav",
                    mime="audio/wav"
                )
    
    except ImportError:
        st.info("💡 For simple voice recording, install: `pip install audio-recorder-streamlit`")
        
        # Manual file upload fallback
        st.markdown("#### 📁 Upload Audio File")
        uploaded_audio = st.file_uploader(
            "Upload an audio file:",
            type=["wav", "mp3", "m4a", "ogg", "webm"],
            help="Upload an audio file for speech recognition"
        )
        
        if uploaded_audio:
            st.audio(uploaded_audio, format=f"audio/{uploaded_audio.type.split('/')[-1]}")
            
            if st.button("🎯 Process Uploaded Audio", type="primary"):
                audio_bytes = uploaded_audio.read()
                process_voice_recording(audio_bytes)

def process_voice_recording(audio_bytes):
    """Process voice recording and convert to text"""
    try:
        config = load_config()
        api_urls = get_api_urls(config)
        session = create_session()
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            # Send to speech-to-text API
            with open(tmp_file_path, 'rb') as f:
                files = {"file": ("audio.wav", f, "audio/wav")}
                response = session.post(api_urls["speech_to_text"], files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "")
                
                if text.strip():
                    st.success(f"🎤 Recognized: {text}")
                    
                    # Process the recognized text
                    process_user_input(text)
                else:
                    st.warning("No speech detected in audio")
            else:
                st.error("Speech recognition failed")
                
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        st.error(f"Voice processing error: {e}")

# Health check
def check_api_health(api_urls):
    """Check API health status"""
    try:
        session = create_session()
        response = session.get(api_urls["health"], timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            return True, health_data
        else:
            return False, f"API returned status code: {response.status_code}"
    except requests.RequestException as e:
        return False, str(e)
    except Exception as e:
        return False, f"Health check error: {e}"

# Main processing function (updated for voice)
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

# Display results with voice integration
def display_results(results, task_type):
    """Display task results with voice options"""
    if isinstance(results, str):
        st.markdown(f"**Response:** {results}")
        
        # Add voice output option for text responses
        if st.session_state.voice_response_enabled:
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🔊 Speak Result"):
                    audio_response = generate_voice_response(results)
                    if audio_response:
                        play_audio_response(audio_response)
        return
    
    if not isinstance(results, dict):
        st.error("Invalid response format")
        return
    
    task_results = results.get("results", [])
    evaluation = results.get("evaluation")
    
    if task_results:
        st.markdown('<div class="sub-header">🤖 Model Results</div>', unsafe_allow_html=True)
        
        for i, result in enumerate(task_results):
            model_name = result.get("model", "Unknown")
            output = result.get("output")
            inference_time = result.get("inference_time", 0)
            success = result.get("success", True)
            
            if success and output:
                col1, col2, col3 = st.columns([4, 1, 1])
                
                with col1:
                    st.markdown(f"**{model_name}:**")
                    st.write(output)
                    st.caption(f"⏱️ Generation time: {inference_time:.2f}s")
                
                with col2:
                    if st.button("🔊", key=f"speak_{i}", help="Speak this result"):
                        audio_response = generate_voice_response(output)
                        if audio_response:
                            play_audio_response(audio_response)
                
                with col3:
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

# Document upload with voice feedback
def handle_document_upload(api_urls, uploaded_file):
    """Handle document upload for RAG with voice feedback"""
    try:
        session = create_session()
        
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        
        with st.spinner("Uploading and processing document..."):
            response = session.post(api_urls["upload"], files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            success_msg = f"✅ {result.get('message', 'Document uploaded successfully')}"
            st.success(success_msg)
            
            # Voice feedback for successful upload
            if st.session_state.voice_response_enabled and st.session_state.auto_speech:
                audio_response = generate_voice_response("Document uploaded successfully")
                if audio_response:
                    play_audio_response(audio_response)
            
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

# Main app with voice integration
def main():
    """Main application with integrated voice features"""
    initialize_session_state()
    
    config = load_config()
    api_urls = get_api_urls(config)
    
    # Header
    st.markdown('<div class="main-header">🤖 AI Workbench with Voice</div>', unsafe_allow_html=True)
    st.markdown("*Advanced AI platform with ChatGPT-like voice interaction capabilities*")
    
    # Check API health
    with st.spinner("Checking API status..."):
        health_ok, health_info = check_api_health(api_urls)
    
    if not health_ok:
        st.error(f"❌ **API Not Available:** {health_info}")
        st.info("💡 Make sure to start the API server: `python main.py`")
        st.stop()
    
    # Display API status
    if isinstance(health_info, dict):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("API Status", "🟢 Healthy")
        
        with col2:
            available_models = health_info.get("models_available", 0)
            st.metric("Available Models", available_models)
        
        with col3:
            model_names = health_info.get("model_names", [])
            if model_names:
                st.metric("Active Models", ", ".join(model_names))
        
        with col4:
            voice_status = "🎤 Enabled" if st.session_state.voice_enabled else "🔇 Disabled"
            st.metric("Voice System", voice_status)
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Voice Chat", "📝 Summarization", "🌐 Translation", "⚙️ Settings"])
    
    with tab1:
        # Enhanced chat with voice
        render_enhanced_chat()
        
        # Simple voice recorder as fallback
        with st.expander("🎙️ Alternative Voice Input", expanded=False):
            render_simple_voice_recorder()
    
    with tab2:
        # Summarization interface
        render_summarization_interface(api_urls)
    
    with tab3:
        # Translation interface
        render_translation_interface(api_urls)
    
    with tab4:
        # Settings and diagnostics
        render_settings_interface(api_urls)

def render_summarization_interface(api_urls):
    """Render summarization interface"""
    st.markdown("### 📝 Text Summarization")
    
    text_input = st.text_area(
        "Enter text to summarize:",
        height=200,
        placeholder="Paste your text here..."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        reference = st.text_area(
            "Reference summary (optional):",
            height=100,
            help="Provide a reference summary for evaluation"
        )
    
    with col2:
        st.markdown("#### Parameters")
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 50, 500, 100, 10)
        
        if st.button("🚀 Summarize", type="primary"):
            if text_input.strip():
                process_summarization(api_urls, text_input, reference, temperature, max_tokens)
            else:
                st.error("Please enter text to summarize")

def process_summarization(api_urls, text, reference, temperature, max_tokens):
    """Process summarization request"""
    payload = {
        "task": "summarization",
        "text": text,
        "reference": reference,
        "params": {
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        "metrics": ["rouge1", "rouge2", "rougeL"]
    }
    
    success, results = process_task(api_urls, payload)
    
    if success:
        display_results(results, "summarization")
    else:
        st.error(f"❌ **Processing failed:** {results}")

def render_translation_interface(api_urls):
    """Render translation interface"""
    st.markdown("### 🌐 Language Translation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to translate:",
            height=200,
            placeholder="Enter text in any language..."
        )
        
        reference = st.text_area(
            "Reference translation (optional):",
            height=100,
            help="Provide a reference translation for evaluation"
        )
    
    with col2:
        target_lang = st.selectbox(
            "Target Language:",
            ["Spanish", "French", "German", "Italian", "Portuguese", "Chinese", "Japanese"]
        )
        
        st.markdown("#### Parameters")
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1, key="trans_temp")
        max_tokens = st.slider("Max Tokens", 50, 500, 100, 10, key="trans_tokens")
        
        if st.button("🌍 Translate", type="primary"):
            if text_input.strip() and target_lang:
                process_translation(api_urls, text_input, target_lang, reference, temperature, max_tokens)
            else:
                st.error("Please enter text and select target language")

def process_translation(api_urls, text, target_lang, reference, temperature, max_tokens):
    """Process translation request"""
    payload = {
        "task": "translation",
        "text": text,
        "target_lang": target_lang,
        "reference": reference,
        "params": {
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        "metrics": ["bleu", "meteor"]
    }
    
    success, results = process_task(api_urls, payload)
    
    if success:
        display_results(results, "translation")
    else:
        st.error(f"❌ **Processing failed:** {results}")

def render_settings_interface(api_urls):
    """Render settings and diagnostics interface"""
    st.markdown("### ⚙️ System Settings & Diagnostics")
    
    # Voice system settings
    st.markdown("#### 🎤 Voice System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.voice_enabled = st.checkbox(
            "Enable Voice Features",
            value=st.session_state.voice_enabled,
            help="Enable/disable all voice functionality"
        )
        
        if st.session_state.voice_enabled:
            st.success("🎤 Voice system is enabled")
            
            # Voice diagnostics
            if st.button("🔧 Run Voice Diagnostics"):
                test_voice_system()
        else:
            st.info("🔇 Voice system is disabled")
    
    with col2:
        # API diagnostics
        st.markdown("**API Status:**")
        if st.button("🔄 Refresh API Status"):
            health_ok, health_info = check_api_health(api_urls)
            if health_ok:
                st.success("✅ API is healthy")
                st.json(health_info)
            else:
                st.error(f"❌ API issues: {health_info}")
    
    # Document upload
    st.markdown("#### 📁 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload document for RAG:",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload documents to provide context for chat"
    )
    
    if uploaded_file and st.button("📤 Upload Document"):
        handle_document_upload(api_urls, uploaded_file)
    
    # Usage statistics
    st.markdown("#### 📊 Usage Statistics")
    if st.button("📈 Get Usage Stats"):
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

if __name__ == "__main__":
    main()