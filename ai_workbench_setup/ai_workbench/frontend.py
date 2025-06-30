import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
import tempfile
import streamlit.components.v1 as components
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Workbench - Enhanced",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional UI
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 0.5rem 0;
    border-left: 5px solid #007bff;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.voice-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 25px;
    margin: 15px 0;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
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
}

.voice-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(0,123,255,0.4);
}

.voice-btn.recording {
    background: linear-gradient(145deg, #dc3545, #c82333);
    animation: pulse-glow 1.5s infinite;
}

@keyframes pulse-glow {
    0% { transform: scale(1); box-shadow: 0 0 20px rgba(220,53,69,0.6); }
    50% { transform: scale(1.05); box-shadow: 0 0 40px rgba(220,53,69,0.8); }
    100% { transform: scale(1); box-shadow: 0 0 20px rgba(220,53,69,0.6); }
}

.voice-status {
    margin: 20px 0;
    padding: 15px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 16px;
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

.success-message { color: #28a745; font-weight: bold; }
.error-message { color: #dc3545; font-weight: bold; }
.warning-message { color: #ffc107; font-weight: bold; }

.stSelectbox > div > div { background-color: rgba(255,255,255,0.9); }
.stTextInput > div > div > input { background-color: rgba(255,255,255,0.9); }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_status" not in st.session_state:
        st.session_state.api_status = None
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = False
    if "voice_input_text" not in st.session_state:
        st.session_state.voice_input_text = ""
    if "last_audio_response" not in st.session_state:
        st.session_state.last_audio_response = None
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None

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
        "text_to_speech": f"{base_url}/text_to_speech",
        "health": f"{base_url}/health",
        "models": f"{base_url}/models",
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

# Enhanced Voice Interface
def render_voice_interface(api_urls):
    """Render voice interface with recording capabilities"""
    
    voice_html = f"""
    <div class="voice-container">
        <h3 style="margin-top: 0;">🎤 Voice Assistant</h3>
        <p style="margin: 10px 0; opacity: 0.9;">
            Click to record, speak naturally, then click again to stop
        </p>
        
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <button id="voiceBtn" class="voice-btn" onclick="toggleRecording()">
                🎤
            </button>
        </div>
        
        <div id="voiceStatus" class="voice-status status-ready">
            Ready to listen - Click the microphone to start
        </div>
        
        <div id="transcription" style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.9); border-radius: 10px; min-height: 50px; color: #333;">
            Your speech will appear here...
        </div>
        
        <div style="margin-top: 15px;">
            <button onclick="clearTranscription()" style="background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3); border-radius: 8px; padding: 8px 15px; color: white; cursor: pointer;">
                🗑️ Clear
            </button>
        </div>
    </div>
    
    <script>
    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];
    
    async function toggleRecording() {{
        const btn = document.getElementById('voiceBtn');
        const status = document.getElementById('voiceStatus');
        
        if (!isRecording) {{
            try {{
                const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = event => {{
                    audioChunks.push(event.data);
                }};
                
                mediaRecorder.onstop = async () => {{
                    const audioBlob = new Blob(audioChunks, {{ type: 'audio/wav' }});
                    await processAudio(audioBlob);
                }};
                
                mediaRecorder.start();
                isRecording = true;
                
                btn.classList.add('recording');
                btn.innerHTML = '🛑';
                status.className = 'voice-status status-recording';
                status.textContent = 'Recording... Click to stop';
                
            }} catch (error) {{
                status.className = 'voice-status status-ready';
                status.textContent = 'Microphone access denied. Please enable permissions.';
                console.error('Error accessing microphone:', error);
            }}
        }} else {{
            mediaRecorder.stop();
            isRecording = false;
            
            btn.classList.remove('recording');
            btn.innerHTML = '⏳';
            status.className = 'voice-status status-processing';
            status.textContent = 'Processing speech...';
            
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }}
    }}
    
    async function processAudio(audioBlob) {{
        const status = document.getElementById('voiceStatus');
        const transcription = document.getElementById('transcription');
        const btn = document.getElementById('voiceBtn');
        
        try {{
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.wav');
            
            const response = await fetch('{api_urls["speech_to_text"]}', {{
                method: 'POST',
                body: formData
            }});
            
            if (response.ok) {{
                const result = await response.json();
                const text = result.text || '';
                
                if (text.trim()) {{
                    transcription.innerHTML = `<strong>You said:</strong> "${{text}}"`;
                    
                    // Send to Streamlit
                    window.parent.postMessage({{
                        type: 'voiceInput',
                        text: text
                    }}, '*');
                    
                }} else {{
                    transcription.innerHTML = '<em>No speech detected. Please try again.</em>';
                }}
            }} else {{
                throw new Error(`HTTP ${{response.status}}`);
            }}
            
        }} catch (error) {{
            console.error('Error processing audio:', error);
            transcription.innerHTML = '<em>Error processing speech. Please try again.</em>';
        }} finally {{
            btn.classList.remove('recording');
            btn.innerHTML = '🎤';
            status.className = 'voice-status status-ready';
            status.textContent = 'Ready for voice input';
        }}
    }}
    
    function clearTranscription() {{
        document.getElementById('transcription').innerHTML = 'Your speech will appear here...';
        window.parent.postMessage({{ type: 'clearTranscription' }}, '*');
    }}
    </script>
    """
    
    components.html(voice_html, height=300)

# Check API health
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

# Main processing function with comprehensive metrics
def process_task_with_metrics(api_urls, payload):
    """Process task with comprehensive metrics collection"""
    try:
        session = create_session()
        
        with st.spinner("Processing with comprehensive metrics..."):
            response = session.post(
                api_urls["process"], 
                json=payload, 
                timeout=120
            )
        
        if response.status_code == 200:
            try:
                result = response.json()
                
                # If we have evaluation data, enhance it with additional metrics
                if "evaluation" in result and result["evaluation"]:
                    enhanced_evaluation = enhance_evaluation_metrics(result["evaluation"])
                    result["evaluation"] = enhanced_evaluation
                
                return True, result
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

def enhance_evaluation_metrics(evaluation_data):
    """Enhance evaluation data with additional computed metrics"""
    try:
        df = pd.DataFrame(evaluation_data)
        
        # Add custom metrics calculations
        for idx, row in df.iterrows():
            if "word_count" in row and "inference_time" in row and row["inference_time"] > 0:
                df.at[idx, "words_per_second"] = row["word_count"] / row["inference_time"]
            
            # Add efficiency score (quality/time tradeoff)
            quality_metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1", "bleu"]
            quality_scores = [row.get(metric, 0) for metric in quality_metrics if metric in row]
            
            if quality_scores and row.get("inference_time", 0) > 0:
                avg_quality = sum(quality_scores) / len(quality_scores)
                df.at[idx, "efficiency_score"] = avg_quality / row["inference_time"]
        
        return df.to_dict('records')
    except Exception as e:
        st.warning(f"Could not enhance metrics: {e}")
        return evaluation_data

# Display comprehensive results
def display_comprehensive_results(results, task_type):
    """Display results with comprehensive metrics and visualizations"""
    if isinstance(results, str):
        st.markdown(f"**Response:** {results}")
        return
    
    if not isinstance(results, dict):
        st.error("Invalid response format")
        return
    
    task_results = results.get("results", [])
    evaluation = results.get("evaluation")
    
    # Display model results
    if task_results:
        st.markdown("### 🤖 Model Results")
        
        tabs = st.tabs([f"📊 Results", f"📈 Metrics", f"🎯 Analysis"])
        
        with tabs[0]:
            for i, result in enumerate(task_results):
                model_name = result.get("model", "Unknown")
                output = result.get("output")
                inference_time = result.get("inference_time", 0)
                success = result.get("success", True)
                
                with st.expander(f"🔍 {model_name} Results", expanded=True):
                    if success and output:
                        st.write(output)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("⏱️ Time", f"{inference_time:.2f}s")
                        with col2:
                            word_count = result.get("word_count", len(output.split()))
                            st.metric("📝 Words", word_count)
                        with col3:
                            if inference_time > 0:
                                wps = word_count / inference_time
                                st.metric("⚡ Speed", f"{wps:.1f} w/s")
                        
                        # Quality indicators
                        quality_issues = result.get("quality_issues", [])
                        if quality_issues:
                            st.warning(f"⚠️ Quality concerns: {', '.join(quality_issues)}")
                        else:
                            st.success("✅ No quality issues detected")
                    else:
                        error_msg = result.get("error", "Unknown error")
                        st.error(f"❌ **Error:** {error_msg}")
        
        with tabs[1]:
            if evaluation and len(evaluation) > 0:
                display_comprehensive_metrics(evaluation)
            else:
                st.info("No evaluation metrics available. Add a reference text for detailed metrics.")
        
        with tabs[2]:
            if evaluation and len(evaluation) > 0:
                display_analysis_insights(evaluation, task_type)
            else:
                st.info("Analysis requires evaluation metrics.")

def display_comprehensive_metrics(evaluation):
    """Display comprehensive metrics with enhanced visualizations"""
    try:
        df = pd.DataFrame(evaluation)
        
        # Metrics overview
        st.markdown("#### 📊 Comprehensive Metrics Overview")
        
        # Metric categories
        metric_categories = {
            "Quality": ["rouge1", "rouge2", "rougeL", "bertscore_f1", "bleu", "meteor"],
            "Performance": ["inference_time", "tokens_per_second", "words_per_second", "efficiency_score"],
            "Content": ["word_count", "sentence_count", "vocabulary_diversity", "compression_ratio"],
            "Advanced": ["coherence_score", "fluency_score", "readability_score", "sentiment_score"]
        }
        
        # Create tabs for different metric categories
        category_tabs = st.tabs(list(metric_categories.keys()) + ["📋 All Metrics"])
        
        for idx, (category, metrics) in enumerate(metric_categories.items()):
            with category_tabs[idx]:
                available_metrics = [m for m in metrics if m in df.columns]
                if available_metrics:
                    display_metric_category(df, category, available_metrics)
                else:
                    st.info(f"No {category.lower()} metrics available")
        
        # All metrics tab
        with category_tabs[-1]:
            st.dataframe(df, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"evaluation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            with col2:
                json_data = df.to_json(orient="records", indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"evaluation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")

def display_metric_category(df, category, metrics):
    """Display metrics for a specific category"""
    st.markdown(f"**{category} Metrics**")
    
    # Summary statistics
    summary_cols = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            with summary_cols[i]:
                values = df[metric].dropna()
                if len(values) > 0:
                    st.metric(
                        label=metric.replace('_', ' ').title(),
                        value=f"{values.mean():.3f}",
                        delta=f"±{values.std():.3f}"
                    )
    
    # Visualizations
    if len(metrics) > 1:
        # Bar chart comparison
        fig = go.Figure()
        for metric in metrics:
            if metric in df.columns:
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=df["model"],
                    y=df[metric],
                    text=df[metric].round(3),
                    textposition="auto"
                ))
        
        fig.update_layout(
            title=f"{category} Metrics Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            template="plotly_white",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual metric details
    for metric in metrics:
        if metric in df.columns:
            with st.expander(f"📋 {metric.replace('_', ' ').title()} Details"):
                metric_df = df[["model", metric]].sort_values(metric, ascending=False)
                st.dataframe(metric_df, use_container_width=True)

def display_analysis_insights(evaluation, task_type):
    """Display analysis insights and recommendations"""
    try:
        df = pd.DataFrame(evaluation)
        
        st.markdown("#### 🎯 Performance Analysis")
        
        # Overall performance summary
        successful_models = df[df.get("success", True) == True]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Models Tested", len(df))
        with col2:
            st.metric("Successful", len(successful_models))
        with col3:
            if "inference_time" in df.columns:
                avg_time = df["inference_time"].mean()
                st.metric("Avg Time", f"{avg_time:.2f}s")
        with col4:
            if "word_count" in df.columns:
                avg_words = df["word_count"].mean()
                st.metric("Avg Words", f"{avg_words:.0f}")
        
        # Best performing model analysis
        if len(successful_models) > 0:
            st.markdown("#### 🏆 Top Performers")
            
            # Determine best model by different criteria
            performance_criteria = {
                "Speed": ("inference_time", "min"),
                "Quality": ("rouge1", "max") if "rouge1" in df.columns else ("word_count", "max"),
                "Efficiency": ("efficiency_score", "max") if "efficiency_score" in df.columns else ("words_per_second", "max")
            }
            
            perf_cols = st.columns(len(performance_criteria))
            
            for i, (criterion, (metric, direction)) in enumerate(performance_criteria.items()):
                with perf_cols[i]:
                    if metric in df.columns:
                        if direction == "max":
                            best_model = df.loc[df[metric].idxmax()]
                        else:
                            best_model = df.loc[df[metric].idxmin()]
                        
                        st.markdown(f"**🥇 Best {criterion}**")
                        st.write(f"**Model:** {best_model['model']}")
                        st.write(f"**Score:** {best_model[metric]:.3f}")
        
        # Recommendations
        st.markdown("#### 💡 Recommendations")
        
        recommendations = generate_recommendations(df, task_type)
        for rec in recommendations:
            st.info(rec)
        
    except Exception as e:
        st.error(f"Error in analysis: {e}")

def generate_recommendations(df, task_type):
    """Generate intelligent recommendations based on results"""
    recommendations = []
    
    try:
        if len(df) == 0:
            return ["No data available for recommendations"]
        
        # Speed recommendations
        if "inference_time" in df.columns:
            avg_time = df["inference_time"].mean()
            if avg_time > 5:
                recommendations.append("⏱️ Consider optimizing for speed - average inference time is high")
            elif avg_time < 1:
                recommendations.append("⚡ Excellent speed performance across models")
        
        # Quality recommendations
        quality_metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1", "bleu"]
        available_quality = [m for m in quality_metrics if m in df.columns]
        
        if available_quality:
            quality_scores = []
            for metric in available_quality:
                quality_scores.extend(df[metric].dropna().tolist())
            
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                if avg_quality < 0.3:
                    recommendations.append("📈 Quality scores are low - consider using reference examples or fine-tuning")
                elif avg_quality > 0.7:
                    recommendations.append("🎯 Excellent quality performance - current approach is working well")
        
        # Model diversity recommendations
        if "model" in df.columns:
            model_performance_variance = {}
            for metric in df.select_dtypes(include=[np.number]).columns:
                if metric != "inference_time":
                    variance = df[metric].var()
                    model_performance_variance[metric] = variance
            
            if model_performance_variance:
                high_variance_metrics = [m for m, v in model_performance_variance.items() if v > 0.1]
                if high_variance_metrics:
                    recommendations.append(f"🔄 High variance in {', '.join(high_variance_metrics)} - consider model ensemble")
        
        # Task-specific recommendations
        if task_type == "summarization":
            if "compression_ratio" in df.columns:
                avg_compression = df["compression_ratio"].mean()
                if avg_compression > 0.8:
                    recommendations.append("📝 Summaries are quite long - consider reducing max_tokens")
                elif avg_compression < 0.2:
                    recommendations.append("📄 Summaries may be too brief - consider increasing min_tokens")
        
        elif task_type == "translation":
            if "bleu" in df.columns:
                avg_bleu = df["bleu"].mean()
                if avg_bleu < 0.2:
                    recommendations.append("🌍 Translation quality is low - try providing more context")
        
        if not recommendations:
            recommendations.append("✅ Performance looks good overall - continue monitoring")
        
    except Exception as e:
        recommendations.append(f"⚠️ Could not generate recommendations: {e}")
    
    return recommendations

# Voice chat interface
def render_voice_chat_interface(api_urls):
    """Render voice-enabled chat interface"""
    st.markdown("### 💬 AI Chat with Voice")
    
    # Voice interface
    if st.session_state.voice_enabled:
        render_voice_interface(api_urls)
        
        # Handle voice input
        if st.session_state.voice_input_text:
            process_chat_input(st.session_state.voice_input_text, api_urls)
            st.session_state.voice_input_text = ""
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Text input
    if prompt := st.chat_input("Type your message or use voice input above..."):
        process_chat_input(prompt, api_urls)

def process_chat_input(text, api_urls):
    """Process chat input (text or voice)"""
    if not text.strip():
        return
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": text})
    
    # Get AI response
    with st.spinner("🤖 AI is thinking..."):
        try:
            session = create_session()
            
            payload = {
                "task": "chat",
                "messages": st.session_state.messages,
                "params": {
                    "temperature": 0.7,
                    "max_tokens": 150,
                    "top_p": 0.9
                }
            }
            
            response = session.post(api_urls["process"], json=payload, timeout=60)
            
            if response.status_code == 200:
                ai_response = response.text.strip()
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
                # Generate voice response if enabled
                if st.session_state.voice_enabled:
                    generate_voice_response(ai_response, api_urls)
            else:
                st.error(f"API Error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.rerun()

def generate_voice_response(text, api_urls):
    """Generate voice response from text"""
    try:
        session = create_session()
        
        payload = {
            "text": text,
            "language": "en",
            "speed": 1.0
        }
        
        response = session.post(api_urls["text_to_speech"], json=payload, timeout=30)
        
        if response.status_code == 200:
            st.session_state.last_audio_response = response.content
            st.audio(response.content, format="audio/mp3", autoplay=True)
        
    except Exception as e:
        st.warning(f"Voice generation failed: {e}")

# Enhanced task interfaces with comprehensive metrics
def render_enhanced_summarization(api_urls):
    """Enhanced summarization interface with comprehensive metrics"""
    st.markdown("### 📝 Advanced Text Summarization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to summarize:",
            height=200,
            placeholder="Paste your text here..."
        )
        
        reference = st.text_area(
            "Reference summary (optional - enables comprehensive metrics):",
            height=100,
            help="Provide a reference summary for detailed evaluation metrics"
        )
    
    with col2:
        st.markdown("#### ⚙️ Parameters")
        temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.7, 0.1)
        max_tokens = st.slider("📏 Max Tokens", 50, 500, 100, 10)
        min_tokens = st.slider("📐 Min Tokens", 10, 200, 30, 5)
        
        st.markdown("#### 📊 Metrics")
        enable_advanced = st.checkbox("Enable Advanced Metrics", value=True)
        
        if enable_advanced:
            selected_metrics = st.multiselect(
                "Select specific metrics:",
                ["rouge1", "rouge2", "rougeL", "bertscore_f1", "coherence_score", 
                 "fluency_score", "compression_ratio", "vocabulary_diversity"],
                default=["rouge1", "rouge2", "rougeL", "bertscore_f1"]
            )
        else:
            selected_metrics = ["rouge1", "rouge2", "rougeL"]
    
    if st.button("🚀 Generate Summary with Comprehensive Analysis", type="primary"):
        if text_input.strip():
            payload = {
                "task": "summarization",
                "text": text_input,
                "reference": reference if reference.strip() else None,
                "params": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "min_tokens": min_tokens
                },
                "metrics": selected_metrics,
                "enable_comprehensive": enable_advanced
            }
            
            success, results = process_task_with_metrics(api_urls, payload)
            
            if success:
                display_comprehensive_results(results, "summarization")
                st.session_state.evaluation_results = results.get("evaluation")
            else:
                st.error(f"❌ **Processing failed:** {results}")
        else:
            st.error("Please enter text to summarize")

def render_enhanced_translation(api_urls):
    """Enhanced translation interface with comprehensive metrics"""
    st.markdown("### 🌐 Advanced Language Translation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to translate:",
            height=200,
            placeholder="Enter text in any language..."
        )
        
        reference = st.text_area(
            "Reference translation (optional - enables comprehensive metrics):",
            height=100,
            help="Provide a reference translation for detailed evaluation"
        )
    
    with col2:
        target_lang = st.selectbox(
            "🎯 Target Language:",
            ["Spanish", "French", "German", "Italian", "Portuguese", 
             "Chinese", "Japanese", "Korean", "Arabic", "Hindi", "Russian"]
        )
        
        st