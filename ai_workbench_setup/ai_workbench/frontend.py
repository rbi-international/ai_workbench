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

# Page configuration
st.set_page_config(
    page_title="AI Workbench",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        # Fallback URLs
        base_url = "http://127.0.0.1:8000"
    else:
        host = config.get("api", {}).get("host", "127.0.0.1")
        port = config.get("api", {}).get("port", 8000)
        base_url = f"http://{host}:{port}"
    
    return {
        "base": base_url,
        "process": f"{base_url}/process",
        "upload": f"{base_url}/upload_documents",
        "voice_input": f"{base_url}/voice_input",
        "voice_output": f"{base_url}/voice_output",
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
            # Handle different response types
            try:
                # Try to parse as JSON first
                return True, response.json()
            except json.JSONDecodeError:
                # If not JSON, return as text (for chat responses)
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
        # Chat response (plain text)
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
            model_name = result.get("model", "Unknown")
            output = result.get("output")
            inference_time = result.get("inference_time", 0)
            success = result.get("success", True)
            
            if success and output:
                st.markdown(f"**{model_name}:**")
                st.write(output)
                st.caption(f"⏱️ Generation time: {inference_time:.2f}s")
                
                # Show additional metrics if available
                word_count = result.get("word_count")
                if word_count:
                    st.caption(f"📝 Words: {word_count}")
                
                # Show quality issues for translation
                quality_issues = result.get("quality_issues", [])
                if quality_issues:
                    st.warning(f"Quality issues: {', '.join(quality_issues)}")
                
            else:
                error_msg = result.get("error", "Unknown error")
                st.error(f"**{model_name}:** {error_msg}")
            
            st.divider()
    
    # Display evaluation metrics
    if evaluation and len(evaluation) > 0:
        st.markdown('<div class="sub-header">📊 Evaluation Metrics</div>', unsafe_allow_html=True)
        
        try:
            eval_df = pd.DataFrame(evaluation)
            st.dataframe(eval_df, use_container_width=True)
            
            # Create visualizations for metrics
            create_metrics_visualization(eval_df, task_type)
            
        except Exception as e:
            st.error(f"Error displaying evaluation: {e}")

def create_metrics_visualization(eval_df, task_type):
    """Create visualizations for evaluation metrics"""
    if eval_df.empty:
        return
    
    try:
        # Determine metrics to plot based on task type
        metric_columns = [col for col in eval_df.columns if col not in ["model", "inference_time"]]
        
        if not metric_columns:
            return
        
        # Create bar chart for metrics
        fig = go.Figure()
        
        for metric in metric_columns:
            if metric in eval_df.columns:
                fig.add_trace(go.Bar(
                    name=metric.upper(),
                    x=eval_df["model"],
                    y=eval_df[metric],
                    text=eval_df[metric].round(3),
                    textposition='auto'
                ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create radar chart if multiple metrics
        if len(metric_columns) > 1:
            create_radar_chart(eval_df, metric_columns)
            
    except Exception as e:
        st.warning(f"Could not create visualization: {e}")

def create_radar_chart(eval_df, metrics):
    """Create radar chart for multi-metric comparison"""
    try:
        fig = go.Figure()
        
        for _, row in eval_df.iterrows():
            values = [row[metric] for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=row["model"]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Multi-Metric Model Comparison",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not create radar chart: {e}")

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
            
            # Show document info
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

# Voice processing
def handle_voice_input(api_urls, voice_file):
    """Handle voice input processing"""
    try:
        session = create_session()
        
        files = {"file": (voice_file.name, voice_file, voice_file.type)}
        
        with st.spinner("Processing voice input..."):
            response = session.post(api_urls["voice_input"], files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("text", "")
            
            if text:
                st.success("✅ Voice input processed successfully")
                return text
            else:
                st.warning("⚠️ No text extracted from voice input")
                return ""
        else:
            st.error("❌ Voice processing failed")
            return ""
            
    except Exception as e:
        st.error(f"❌ Voice processing error: {e}")
        return ""

# Crowdsourcing
def handle_crowdsource_submission(api_urls, data, submitter):
    """Handle crowdsource dataset submission"""
    try:
        session = create_session()
        
        payload = {"data": data, "submitter": submitter}
        
        response = session.post(api_urls["crowdsource"], json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            dataset_id = result.get("dataset_id")
            st.success(f"✅ Dataset submitted with ID: {dataset_id}")
        else:
            st.error("❌ Dataset submission failed")
            
    except Exception as e:
        st.error(f"❌ Crowdsource submission error: {e}")

# Main app
def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()
    
    # Load configuration
    config = load_config()
    api_urls = get_api_urls(config)
    
    # Header
    st.markdown('<div class="main-header">🤖 AI Workbench</div>', unsafe_allow_html=True)
    st.markdown("*A comprehensive AI platform for text processing, translation, and chat*")
    
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
            ["summarization", "translation", "chat"],
            format_func=lambda x: {
                "summarization": "📝 Text Summarization",
                "translation": "🌐 Language Translation", 
                "chat": "💬 AI Chat"
            }[x]
        )
        
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
            st.markdown("### 💬 AI Chat")
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Type your message..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Prepare for API call
                text_input = prompt
                messages = st.session_state.messages
    
    with col2:
        # Sidebar controls
        st.markdown('<div class="sub-header">⚙️ Parameters</div>', unsafe_allow_html=True)
        
        # Model parameters
        with st.expander("🎛️ Model Settings", expanded=True):
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
        
        # Voice input
        with st.expander("🎤 Voice Input"):
            voice_file = st.file_uploader(
                "Upload audio file:",
                type=["wav", "mp3", "m4a"],
                help="Upload an audio file to convert to text"
            )
            
            if voice_file:
                if st.button("🎯 Process Voice"):
                    voice_text = handle_voice_input(api_urls, voice_file)
                    if voice_text:
                        if task in ["summarization", "translation"]:
                            text_input = voice_text
                            st.success(f"Voice input: {voice_text[:100]}...")
                        elif task == "chat":
                            st.session_state.messages.append({"role": "user", "content": voice_text})
                            st.rerun()
        
        # Document upload for RAG
        with st.expander("📁 Document Upload"):
            uploaded_file = st.file_uploader(
                "Upload document:",
                type=["pdf", "png", "jpg", "jpeg"],
                help="Upload documents for context-aware responses"
            )
            
            if uploaded_file and st.button("📤 Upload"):
                handle_document_upload(api_urls, uploaded_file)
    
    # Process button and results
    st.markdown("---")
    
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
                    # Display structured results
                    display_results(results, task)
            else:
                st.error(f"❌ **Processing failed:** {results}")
    
    # Footer with additional features
    st.markdown("---")
    
    # Additional features in expandable sections
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
                        handle_crowdsource_submission(api_urls, data, submitter)
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

if __name__ == "__main__":
    main()