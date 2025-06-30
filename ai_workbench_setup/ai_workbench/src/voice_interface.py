# voice_interface.py - Add this file to handle voice interactions

import streamlit as st
import streamlit.components.v1 as components

def render_voice_interface():
    """Render voice interface with recording capabilities"""
    
    # HTML and JavaScript for voice recording
    voice_html = """
    <div id="voice-container">
        <style>
        .voice-btn {
            background: #007bff;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            color: white;
            font-size: 24px;
            cursor: pointer;
            margin: 10px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .voice-btn:hover {
            background: #0056b3;
            transform: scale(1.05);
        }
        .voice-btn.recording {
            background: #dc3545;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .voice-status {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .status-ready { background: #d4edda; color: #155724; }
        .status-recording { background: #f8d7da; color: #721c24; }
        .status-processing { background: #fff3cd; color: #856404; }
        </style>
        
        <div style="text-align: center;">
            <button id="voiceBtn" class="voice-btn" onclick="toggleRecording()">
                🎤
            </button>
            <div id="voiceStatus" class="voice-status status-ready">
                Click the microphone to start voice input
            </div>
            <div id="transcription" style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; min-height: 40px;">
                Your speech will appear here...
            </div>
        </div>
        
        <script>
        let isRecording = false;
        let mediaRecorder;
        let audioChunks = [];
        
        async function toggleRecording() {
            const btn = document.getElementById('voiceBtn');
            const status = document.getElementById('voiceStatus');
            
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = event => {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        await processAudio(audioBlob);
                    };
                    
                    mediaRecorder.start();
                    isRecording = true;
                    
                    btn.classList.add('recording');
                    btn.innerHTML = '🛑';
                    status.className = 'voice-status status-recording';
                    status.textContent = 'Recording... Click to stop';
                    
                } catch (error) {
                    status.className = 'voice-status status-ready';
                    status.textContent = 'Microphone access denied. Please enable microphone permissions.';
                    console.error('Error accessing microphone:', error);
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                
                btn.classList.remove('recording');
                btn.innerHTML = '🎤';
                status.className = 'voice-status status-processing';
                status.textContent = 'Processing speech...';
                
                // Stop all tracks
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        }
        
        async function processAudio(audioBlob) {
            const status = document.getElementById('voiceStatus');
            const transcription = document.getElementById('transcription');
            
            try {
                // Convert blob to base64 for transmission
                const reader = new FileReader();
                reader.onloadend = function() {
                    const base64Audio = reader.result.split(',')[1];
                    
                    // Send to Streamlit via session state
                    window.parent.postMessage({
                        type: 'voiceInput',
                        audio: base64Audio,
                        timestamp: Date.now()
                    }, '*');
                };
                reader.readAsDataURL(audioBlob);
                
                status.className = 'voice-status status-ready';
                status.textContent = 'Speech sent for processing...';
                
            } catch (error) {
                status.className = 'voice-status status-ready';
                status.textContent = 'Error processing speech. Please try again.';
                console.error('Error processing audio:', error);
            }
        }
        
        // Listen for messages from Streamlit
        window.addEventListener('message', function(event) {
            if (event.data.type === 'transcriptionResult') {
                const transcription = document.getElementById('transcription');
                const status = document.getElementById('voiceStatus');
                
                if (event.data.text) {
                    transcription.textContent = event.data.text;
                    status.className = 'voice-status status-ready';
                    status.textContent = 'Speech recognized! Click microphone for more input.';
                } else {
                    transcription.textContent = 'Could not understand speech. Please try again.';
                    status.className = 'voice-status status-ready';
                    status.textContent = 'Ready for voice input';
                }
            }
        });
        </script>
    </div>
    """
    
    # Render the voice interface
    components.html(voice_html, height=200)

def handle_voice_message():
    """Handle voice messages from the JavaScript component"""
    # This would be called when voice input is received
    # In practice, you'd integrate this with your speech-to-text API
    pass

# Example integration in your chat interface
def enhanced_chat_interface():
    """Enhanced chat interface with voice support"""
    
    st.markdown("### 💬 AI Chat with Voice")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Voice interface
    with st.expander("🎤 Voice Input", expanded=True):
        render_voice_interface()
        
        # Text fallback
        st.markdown("**Or type your message:**")
        text_input = st.text_input("Message:", key="text_fallback")
        
        if text_input:
            # Process text input
            st.session_state.messages.append({"role": "user", "content": text_input})
            st.rerun()
    
    # Regular chat input as fallback
    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

# Streamlit WebRTC alternative for more robust voice handling
def setup_webrtc_voice():
    """Setup WebRTC for voice input (requires streamlit-webrtc)"""
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        import av
        import numpy as np
        
        def audio_frame_callback(frame):
            # Process audio frame
            audio_array = frame.to_ndarray()
            # Store audio data for processing
            if "audio_buffer" not in st.session_state:
                st.session_state.audio_buffer = []
            st.session_state.audio_buffer.append(audio_array)
            return frame
        
        # WebRTC streamer for voice input
        webrtc_ctx = webrtc_streamer(
            key="voice-input",
            mode=WebRtcMode.SENDONLY,
            audio_frame_callback=audio_frame_callback,
            media_stream_constraints={
                "video": False,
                "audio": {
                    "echoCancellation": True,
                    "noiseSuppression": True,
                    "autoGainControl": True,
                }
            }
        )
        
        return webrtc_ctx
        
    except ImportError:
        st.warning("streamlit-webrtc not installed. Install with: pip install streamlit-webrtc")
        return None

# Simple voice recorder using audio-recorder-streamlit
def simple_voice_recorder():
    """Simple voice recorder using audio-recorder-streamlit package"""
    try:
        from audio_recorder_streamlit import audio_recorder
        
        # Record audio
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="2x"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            return audio_bytes
        
        return None
        
    except ImportError:
        st.info("For better voice recording, install: pip install audio-recorder-streamlit")
        return None

# Integration helper functions
def process_voice_to_chat(api_urls, audio_data, messages, params):
    """Process voice input and add to chat"""
    try:
        import requests
        import tempfile
        import os
        
        if not audio_data:
            return False
        
        # Save audio data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        
        try:
            # Send to speech-to-text API
            with open(tmp_file_path, 'rb') as f:
                files = {"file": ("audio.wav", f, "audio/wav")}
                response = requests.post(api_urls["speech_to_text"], files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "")
                
                if text.strip():
                    # Add user message to chat
                    st.session_state.messages.append({"role": "user", "content": text})
                    st.success(f"🎤 Voice input: {text}")
                    return True
                else:
                    st.warning("No speech detected in audio")
                    return False
            else:
                st.error("Speech recognition failed")
                return False
                
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        st.error(f"Voice processing error: {e}")
        return False

def create_voice_enabled_chat():
    """Create a complete voice-enabled chat interface"""
    
    st.markdown("### 💬 Voice-Enabled AI Chat")
    
    # Initialize API URLs
    config = load_config()
    api_urls = get_api_urls(config)
    
    # Voice recording section
    st.markdown("#### 🎤 Voice Input")
    
    # Try different voice input methods
    voice_method = st.selectbox(
        "Voice Input Method:",
        ["Simple Recorder", "Browser API", "Manual Upload"],
        help="Choose your preferred voice input method"
    )
    
    audio_data = None
    
    if voice_method == "Simple Recorder":
        # Use audio-recorder-streamlit if available
        audio_data = simple_voice_recorder()
        
    elif voice_method == "Browser API":
        # Use custom HTML/JS interface
        st.markdown("**Browser-based voice recording:**")
        render_voice_interface()
        
        # Check for voice input in session state
        if "voice_input_text" in st.session_state and st.session_state.voice_input_text:
            voice_text = st.session_state.voice_input_text
            st.session_state.messages.append({"role": "user", "content": voice_text})
            st.session_state.voice_input_text = ""  # Clear after use
            st.rerun()
            
    elif voice_method == "Manual Upload":
        # File upload fallback
        uploaded_audio = st.file_uploader(
            "Upload audio file:",
            type=["wav", "mp3", "m4a", "ogg"],
            help="Upload an audio file for speech recognition"
        )
        
        if uploaded_audio:
            audio_data = uploaded_audio.read()
    
    # Process voice input
    if audio_data and st.button("🎯 Process Voice Input"):
        if process_voice_to_chat(api_urls, audio_data, st.session_state.messages, {}):
            st.rerun()
    
    # Display chat history
    st.markdown("#### 💬 Conversation")
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Text input fallback
    st.markdown("#### ⌨️ Text Input")
    if prompt := st.chat_input("Type your message or use voice input above..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response (you'll need to implement this based on your API)
        # This is a placeholder - replace with actual API call
        ai_response = get_ai_response(api_urls, st.session_state.messages)
        
        if ai_response:
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.write(ai_response)
        
        st.rerun()

def get_ai_response(api_urls, messages):
    """Get AI response from your chat API"""
    try:
        import requests
        
        payload = {
            "task": "chat",
            "messages": messages,
            "params": {
                "temperature": 0.7,
                "max_tokens": 150
            }
        }
        
        response = requests.post(api_urls["process"], json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.text  # Assuming the API returns the response directly
        else:
            return "Sorry, I couldn't process your request right now."
            
    except Exception as e:
        return f"Error: {str(e)}"

# Main integration function for your frontend
def integrate_voice_chat():
    """Main function to integrate voice chat into your existing frontend"""
    
    # Add this to your main frontend file where you handle chat
    if "voice_chat_enabled" not in st.session_state:
        st.session_state.voice_chat_enabled = False
    
    # Toggle for voice features
    st.session_state.voice_chat_enabled = st.checkbox(
        "🎤 Enable Voice Chat",
        value=st.session_state.voice_chat_enabled,
        help="Enable voice input and output for natural conversation"
    )
    
    if st.session_state.voice_chat_enabled:
        create_voice_enabled_chat()
    else:
        # Your existing chat interface
        st.info("Voice chat is disabled. Enable it above for voice interaction.")

# Quick setup instructions
def show_voice_setup_instructions():
    """Show setup instructions for voice features"""
    
    st.markdown("## 🎤 Voice Chat Setup")
    
    st.markdown("""
    ### Installation Options:
    
    **Option 1: Simple Audio Recorder (Recommended)**
    ```bash
    pip install audio-recorder-streamlit
    ```
    
    **Option 2: Advanced WebRTC (More Features)**
    ```bash
    pip install streamlit-webrtc
    ```
    
    **Option 3: Browser-based (No additional packages)**
    - Uses HTML5 Web Audio API
    - Works with modern browsers
    - May require HTTPS for production
    
    ### Usage:
    1. Choose your preferred voice input method
    2. Click the microphone button to record
    3. Speak your message clearly
    4. Click stop or the button again to finish
    5. The AI will respond with text (and optionally voice)
    
    ### Troubleshooting:
    - **Microphone not working**: Check browser permissions
    - **No audio detected**: Ensure microphone is not muted
    - **Recognition errors**: Speak clearly and check internet connection
    """)

# Export the main functions for use in your frontend
__all__ = [
    'render_voice_interface',
    'create_voice_enabled_chat',
    'integrate_voice_chat',
    'simple_voice_recorder',
    'process_voice_to_chat',
    'show_voice_setup_instructions'
]