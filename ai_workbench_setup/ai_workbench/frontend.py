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
import tempfile
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Workbench",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS styling inspired by ChatGPT/Claude
st.markdown("""
<style>
/* Import modern font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global styles */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Professional header */
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1a1a1a;
    text-align: center;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sub-header {
    font-size: 1.1rem;
    color: #6b7280;
    text-align: center;
    margin-bottom: 3rem;
    font-weight: 400;
}

/* Chat container styling */
.chat-container {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
    overflow: hidden;
}

/* Message styling */
.user-message {
    background: #f3f4f6;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 4px solid #3b82f6;
}

.assistant-message {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 4px solid #10b981;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

/* Input styling */
.stTextInput > div > div > input {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    padding: 0.75rem 1rem;
    font-size: 1rem;
    transition: border-color 0.2s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 500;
    font-size: 1rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

/* Sidebar styling */
.css-1d391kg {
    background: #f8fafc;
    border-right: 1px solid #e5e7eb;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #f8fafc;
    border-radius: 8px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    font-weight: 500;
    padding: 0.5rem 1rem;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: white;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Card styling */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    margin: 0.5rem 0;
    transition: box-shadow 0.2s ease;
}

.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* Success/Error/Warning messages */
.stSuccess {
    background: #ecfdf5;
    border: 1px solid #10b981;
    border-radius: 8px;
    color: #065f46;
}

.stError {
    background: #fef2f2;
    border: 1px solid #ef4444;
    border-radius: 8px;
    color: #991b1b;
}

.stWarning {
    background: #fffbeb;
    border: 1px solid #f59e0b;
    border-radius: 8px;
    color: #92400e;
}

/* Loading spinner */
.stSpinner {
    text-align: center;
    color: #667eea;
}

/* Selectbox styling */
.stSelectbox > div > div {
    border-radius: 8px;
    border: 1px solid #d1d5db;
}

/* Text area styling */
.stTextArea > div > div > textarea {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    font-family: 'Inter', sans-serif;
}

/* Feature grid */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.feature-card {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    border: 1px solid #e5e7eb;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
}

.feature-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
}

.feature-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 0.5rem;
}

.feature-description {
    color: #6b7280;
    line-height: 1.6;
}

/* Chat input styling */
.chat-input-container {
    position: sticky;
    bottom: 0;
    background: white;
    padding: 1rem;
    border-top: 1px solid #e5e7eb;
    border-radius: 12px 12px 0 0;
}

/* Model selector */
.model-selector {
    background: #f8fafc;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid #e5e7eb;
}

/* Results container */
.results-container {
    background: white;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    margin: 1rem 0;
    overflow: hidden;
}

.result-header {
    background: #f8fafc;
    padding: 1rem;
    border-bottom: 1px solid #e5e7eb;
    font-weight: 600;
    color: #374151;
}

.result-content {
    padding: 1.5rem;
}

/* Responsive design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem;
    }
    
    .feature-grid {
        grid-template-columns: 1fr;
    }
    
    .main-header {
        font-size: 2rem;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .metric-card, .chat-container, .results-container {
        background: #1f2937;
        border-color: #374151;
        color: #f9fafb;
    }
    
    .feature-card {
        background: #1f2937;
        border-color: #374151;
        color: #f9fafb;
    }
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
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"

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

# Main processing function
def process_task_with_metrics(api_urls, payload):
    """Process task with comprehensive metrics collection"""
    try:
        session = create_session()
        
        with st.spinner("🤖 Processing with AI models..."):
            response = session.post(
                api_urls["process"], 
                json=payload, 
                timeout=120
            )
        
        if response.status_code == 200:
            try:
                result = response.json()
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

# Professional Chat Interface
def render_chat_interface(api_urls):
    """Render professional chat interface"""
    
    # Chat header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-header">AI Assistant</h1>
        <p class="sub-header">Powered by advanced language models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            available_models = get_available_models(api_urls)
            if available_models:
                selected_model = st.selectbox(
                    "Choose AI Model:",
                    available_models,
                    key="model_selector",
                    help="Select which AI model to use for responses"
                )
            else:
                st.error("No models available")
                return

    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 AI Assistant</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    with st.container():
        prompt = st.chat_input("Type your message here...", key="chat_input")
        
        if prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get AI response
            with st.spinner("🤖 Thinking..."):
                try:
                    payload = {
                        "task": "chat",
                        "messages": st.session_state.messages,
                        "params": {
                            "temperature": 0.7,
                            "max_tokens": 200,
                            "top_p": 0.9
                        }
                    }
                    
                    success, result = process_task_with_metrics(api_urls, payload)
                    
                    if success:
                        if isinstance(result, str):
                            ai_response = result
                        else:
                            ai_response = result.get("response", "No response generated")
                        
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        st.rerun()
                    else:
                        st.error(f"Error: {result}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")

def get_available_models(api_urls):
    """Get available models from API"""
    try:
        session = create_session()
        response = session.get(api_urls["models"], timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]
        return ["gpt-4o"]  # Default fallback
    except:
        return ["gpt-4o"]  # Default fallback

# Display results function
def display_comprehensive_results(results, task_type):
    """Display results with comprehensive metrics and visualizations"""
    if isinstance(results, str):
        st.markdown(f"""
        <div class="results-container">
            <div class="result-header">AI Response</div>
            <div class="result-content">{results}</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if not isinstance(results, dict):
        st.error("Invalid response format")
        return
    
    task_results = results.get("results", [])
    evaluation = results.get("evaluation")
    
    # Display model results
    if task_results:
        st.markdown("### 🤖 Model Results")
        
        tabs = st.tabs(["📊 Results", "📈 Metrics", "🎯 Analysis"])
        
        with tabs[0]:
            for i, result in enumerate(task_results):
                model_name = result.get("model", "Unknown")
                output = result.get("output")
                inference_time = result.get("inference_time", 0)
                success = result.get("success", True)
                
                with st.expander(f"🔍 {model_name} Results", expanded=True):
                    if success and output:
                        st.markdown(f"""
                        <div class="results-container">
                            <div class="result-content">{output}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
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
                display_metrics_tab(evaluation)
            else:
                st.info("No evaluation metrics available. Add a reference text for detailed metrics.")
        
        with tabs[2]:
            if evaluation and len(evaluation) > 0:
                display_analysis_tab(evaluation, task_type)
            else:
                st.info("Analysis requires evaluation metrics.")

def display_metrics_tab(evaluation):
    """Display metrics in organized tabs"""
    try:
        df = pd.DataFrame(evaluation)
        
        # Create metric categories
        metric_categories = {
            "Quality": ["rouge1", "rouge2", "rougeL", "bertscore_f1", "bleu", "meteor"],
            "Performance": ["inference_time", "tokens_per_second", "words_per_second"],
            "Content": ["word_count", "sentence_count", "vocabulary_diversity", "compression_ratio"]
        }
        
        category_tabs = st.tabs(list(metric_categories.keys()) + ["📋 All Metrics"])
        
        for idx, (category, metrics) in enumerate(metric_categories.items()):
            with category_tabs[idx]:
                available_metrics = [m for m in metrics if m in df.columns]
                if available_metrics:
                    # Summary statistics
                    cols = st.columns(len(available_metrics))
                    for i, metric in enumerate(available_metrics):
                        with cols[i]:
                            values = df[metric].dropna()
                            if len(values) > 0:
                                st.metric(
                                    label=metric.replace('_', ' ').title(),
                                    value=f"{values.mean():.3f}",
                                    delta=f"±{values.std():.3f}"
                                )
                    
                    # Visualization
                    if len(available_metrics) > 1:
                        fig = go.Figure()
                        for metric in available_metrics:
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
                else:
                    st.info(f"No {category.lower()} metrics available")
        
        # All metrics tab
        with category_tabs[-1]:
            st.dataframe(df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")

def display_analysis_tab(evaluation, task_type):
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
            
            performance_criteria = {
                "Speed": ("inference_time", "min"),
                "Quality": ("rouge1", "max") if "rouge1" in df.columns else ("word_count", "max"),
                "Efficiency": ("words_per_second", "max") if "words_per_second" in df.columns else ("inference_time", "min")
            }
            
            perf_cols = st.columns(len(performance_criteria))
            
            for i, (criterion, (metric, direction)) in enumerate(performance_criteria.items()):
                with perf_cols[i]:
                    if metric in df.columns:
                        if direction == "max":
                            best_model = df.loc[df[metric].idxmax()]
                        else:
                            best_model = df.loc[df[metric].idxmin()]
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🥇 Best {criterion}</h4>
                            <p><strong>Model:</strong> {best_model['model']}</p>
                            <p><strong>Score:</strong> {best_model[metric]:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error in analysis: {e}")

# Summarization Interface
def render_summarization_interface(api_urls):
    """Professional summarization interface"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-header">📝 Text Summarization</h1>
        <p class="sub-header">Generate concise summaries from your text</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to summarize:",
            height=200,
            placeholder="Paste your text here...",
            help="Enter the text you want to summarize"
        )
        
        reference = st.text_area(
            "Reference summary (optional):",
            height=100,
            help="Provide a reference summary for detailed evaluation metrics"
        )
    
    with col2:
        st.markdown("#### ⚙️ Parameters")
        temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.7, 0.1, help="Controls randomness in generation")
        max_tokens = st.slider("📏 Max Tokens", 50, 500, 100, 10, help="Maximum length of summary")
        min_tokens = st.slider("📐 Min Tokens", 10, 200, 30, 5, help="Minimum length of summary")
    
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if text_input.strip():
            payload = {
                "task": "summarization",
                "text": text_input,
                "reference": reference if reference.strip() else None,
                "params": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "min_tokens": min_tokens
                }
            }
            
            success, results = process_task_with_metrics(api_urls, payload)
            
            if success:
                display_comprehensive_results(results, "summarization")
                st.session_state.evaluation_results = results.get("evaluation")
            else:
                st.error(f"❌ **Processing failed:** {results}")
        else:
            st.error("Please enter text to summarize")

# Translation Interface
def render_translation_interface(api_urls):
    """Professional translation interface"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-header">🌐 Language Translation</h1>
        <p class="sub-header">Translate text between languages</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to translate:",
            height=200,
            placeholder="Enter text in any language...",
            help="Enter the text you want to translate"
        )
        
        reference = st.text_area(
            "Reference translation (optional):",
            height=100,
            help="Provide a reference translation for detailed evaluation"
        )
    
    with col2:
        target_lang = st.selectbox(
            "🎯 Target Language:",
            ["Spanish", "French", "German", "Italian", "Portuguese", 
             "Chinese", "Japanese", "Korean", "Arabic", "Hindi", "Russian"],
            help="Select the target language for translation"
        )
        
        temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.3, 0.1, help="Controls randomness in translation")
        max_tokens = st.slider("📏 Max Tokens", 50, 500, 200, 10, help="Maximum length of translation")
    
    if st.button("🌐 Translate Text", type="primary", use_container_width=True):
        if text_input.strip():
            payload = {
                "task": "translation",
                "text": text_input,
                "target_lang": target_lang,
                "reference": reference if reference.strip() else None,
                "params": {
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            }
            
            success, results = process_task_with_metrics(api_urls, payload)
            
            if success:
                display_comprehensive_results(results, "translation")
                st.session_state.evaluation_results = results.get("evaluation")
            else:
                st.error(f"❌ **Processing failed:** {results}")
        else:
            st.error("Please enter text to translate")

# Document Upload Interface
def render_document_interface(api_urls):
    """Document upload and processing interface"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-header">📄 Document Intelligence</h1>
        <p class="sub-header">Upload and analyze your documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, images, text files)",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=True,
        help="Upload documents to ask questions about their content"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file not in st.session_state.uploaded_documents:
                # Process and upload the file
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        session = create_session()
                        response = session.post(api_urls["upload"], files=files, timeout=60)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                            st.session_state.uploaded_documents.append(uploaded_file.name)
                        else:
                            st.error(f"❌ Failed to upload {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error uploading {uploaded_file.name}: {e}")
    
    # Display uploaded documents
    if st.session_state.uploaded_documents:
        st.markdown("#### 📚 Uploaded Documents")
        for doc in st.session_state.uploaded_documents:
            st.markdown(f"""
            <div class="metric-card">
                📄 {doc}
            </div>
            """, unsafe_allow_html=True)
    
    # Document query interface
    st.markdown("#### ❓ Ask Questions About Your Documents")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic of the documents?",
        help="Ask questions about the content of your uploaded documents"
    )
    
    if st.button("🔍 Search Documents", use_container_width=True) and question:
        with st.spinner("Searching through documents..."):
            try:
                # Use chat endpoint with document context
                payload = {
                    "task": "chat",
                    "text": question,
                    "params": {
                        "temperature": 0.3,
                        "max_tokens": 200
                    }
                }
                
                session = create_session()
                response = session.post(api_urls["process"], json=payload, timeout=60)
                
                if response.status_code == 200:
                    answer = response.text
                    
                    st.markdown("#### 💡 Answer")
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>Question:</strong> {question}
                    </div>
                    <div class="assistant-message">
                        <strong>Answer:</strong> {answer}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Failed to process document query")
                    
            except Exception as e:
                st.error(f"Error processing query: {e}")

# Analytics Interface
def render_analytics_interface(api_urls):
    """Analytics and system monitoring interface"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-header">📈 System Analytics</h1>
        <p class="sub-header">Monitor performance and usage statistics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System health check
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh System Status", use_container_width=True):
            with st.spinner("Checking system health..."):
                is_healthy, health_data = check_api_health(api_urls)
                st.session_state.api_status = health_data
    
    with col2:
        if st.button("📊 Get Usage Statistics", use_container_width=True):
            with st.spinner("Fetching usage data..."):
                try:
                    session = create_session()
                    response = session.get(api_urls["usage"], timeout=10)
                    if response.status_code == 200:
                        usage_data = response.json()
                        st.session_state.usage_stats = usage_data
                except Exception as e:
                    st.error(f"Failed to fetch usage stats: {e}")
    
    # Display system status
    if st.session_state.api_status:
        st.markdown("#### 🚀 System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = st.session_state.api_status.get("status", "unknown")
            if status == "healthy":
                st.markdown("""
                <div class="metric-card">
                    <h4>API Status</h4>
                    <p style="color: #10b981;">✅ Healthy</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <h4>API Status</h4>
                    <p style="color: #ef4444;">❌ Issues</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            models_available = st.session_state.api_status.get("models_available", 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Models Available</h4>
                <p>{models_available}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            components = st.session_state.api_status.get("components", {})
            active_components = sum(1 for comp in components.values() if comp)
            st.markdown(f"""
            <div class="metric-card">
                <h4>Active Components</h4>
                <p>{active_components}/{len(components)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            timestamp = datetime.fromtimestamp(st.session_state.api_status.get("timestamp", 0))
            st.markdown(f"""
            <div class="metric-card">
                <h4>Last Check</h4>
                <p>{timestamp.strftime('%H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)

# Navigation
def render_navigation():
    """Render professional navigation"""
    st.markdown("""
    <div style="background: white; padding: 1rem; border-bottom: 1px solid #e5e7eb; margin-bottom: 2rem;">
        <div style="max-width: 1200px; margin: 0 auto;">
    """, unsafe_allow_html=True)
    
    # Navigation tabs
    tabs = st.tabs(["💬 Chat", "📝 Summarization", "🌐 Translation", "📄 Documents", "📈 Analytics"])
    
    return tabs

# Main Application
def main():
    """Main application entry point"""
    initialize_session_state()
    
    # Load configuration
    config = load_config()
    api_urls = get_api_urls(config)
    
    # Navigation
    tabs = render_navigation()
    
    # Tab content
    with tabs[0]:  # Chat
        render_chat_interface(api_urls)
    
    with tabs[1]:  # Summarization
        render_summarization_interface(api_urls)
    
    with tabs[2]:  # Translation
        render_translation_interface(api_urls)
    
    with tabs[3]:  # Documents
        render_document_interface(api_urls)
    
    with tabs[4]:  # Analytics
        render_analytics_interface(api_urls)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem; margin-top: 3rem; border-top: 1px solid #e5e7eb;">
        <p>🤖 AI Workbench - Professional AI Platform</p>
        <p style="font-size: 0.875rem;">Powered by OpenAI and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()