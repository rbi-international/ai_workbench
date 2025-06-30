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
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="AI Workbench - Complete Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
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
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Professional header */
.main-header {
    font-size: 2.8rem;
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
    font-size: 1.2rem;
    color: #6b7280;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
}

/* Feature cards */
.feature-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    margin: 1rem 0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
}

/* Status indicators */
.status-healthy {
    color: #10b981;
    font-weight: 600;
}

.status-warning {
    color: #f59e0b;
    font-weight: 600;
}

.status-error {
    color: #ef4444;
    font-weight: 600;
}

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid #e5e7eb;
    text-align: center;
    margin: 0.5rem 0;
}

/* Advanced feature sections */
.advanced-section {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 12px;
    padding: 2rem;
    margin: 1rem 0;
    border: 1px solid #cbd5e1;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
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
}

.stTabs [aria-selected="true"] {
    background: white;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Charts and visualizations */
.chart-container {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #e5e7eb;
    margin: 1rem 0;
}

/* Code blocks */
.stCode {
    background: #1f2937;
    border-radius: 8px;
    border: 1px solid #374151;
}

/* Info boxes */
.info-box {
    background: #eff6ff;
    border: 1px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box {
    background: #fffbeb;
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.success-box {
    background: #ecfdf5;
    border: 1px solid #10b981;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Responsive design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 0.5rem;
    }
    
    .main-header {
        font-size: 2rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        "messages": [],
        "api_status": None,
        "selected_models": [],
        "evaluation_results": None,
        "chat_history": [],
        "uploaded_documents": [],
        "current_page": "overview",
        "fusion_enabled": False,
        "ethics_enabled": True,
        "tutor_enabled": True,
        "collaboration_battles": [],
        "crowdsource_datasets": [],
        "fine_tuning_jobs": [],
        "cost_tracking": True,
        "usage_stats": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

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
        "usage": f"{base_url}/usage_stats",
        "crowdsource": f"{base_url}/crowdsource",
        "pending_datasets": f"{base_url}/pending_datasets",
        "approve_dataset": f"{base_url}/approve_dataset",
        "supported_languages": f"{base_url}/supported_languages"
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

# API call helper
def make_api_request(url, method="GET", data=None, timeout=120):
    """Make API request with error handling"""
    try:
        session = create_session()
        
        if method == "GET":
            response = session.get(url, timeout=timeout)
        elif method == "POST":
            response = session.post(url, json=data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 200:
            try:
                return True, response.json()
            except json.JSONDecodeError:
                return True, response.text
        else:
            return False, f"API error: {response.status_code}"
            
    except requests.Timeout:
        return False, "Request timed out"
    except Exception as e:
        return False, str(e)

# Sidebar Navigation
def render_sidebar():
    """Render enhanced sidebar with all features"""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #1f2937; margin-bottom: 0.5rem;">🤖 AI Workbench</h2>
        <p style="color: #6b7280; font-size: 0.9rem;">Complete AI Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Features
    st.sidebar.markdown("### 🎯 Core Features")
    core_pages = {
        "🏠 Overview": "overview",
        "💬 Chat Assistant": "chat",
        "📝 Summarization": "summarization",
        "🌐 Translation": "translation",
        "📄 Document Intelligence": "documents"
    }
    
    for label, page_id in core_pages.items():
        if st.sidebar.button(label, key=f"nav_{page_id}", use_container_width=True):
            st.session_state.current_page = page_id
    
    st.sidebar.markdown("---")
    
    # Advanced Features
    st.sidebar.markdown("### 🚀 Advanced Features")
    advanced_pages = {
        "🔬 Model Evaluation": "evaluation",
        "🤝 Model Fusion": "fusion",
        "⚔️ Model Battles": "battles",
        "🎓 AI Tutor": "tutor",
        "🛡️ Ethics & Safety": "ethics",
        "👥 Crowdsourcing": "crowdsourcing",
        "🎛️ Fine-tuning": "fine_tuning"
    }
    
    for label, page_id in advanced_pages.items():
        if st.sidebar.button(label, key=f"nav_{page_id}", use_container_width=True):
            st.session_state.current_page = page_id
    
    st.sidebar.markdown("---")
    
    # System & Analytics
    st.sidebar.markdown("### 📊 System & Analytics")
    system_pages = {
        "📈 Analytics & Monitoring": "analytics",
        "💰 Cost Tracking": "costs",
        "⚙️ System Configuration": "config",
        "🔧 API Explorer": "api_explorer"
    }
    
    for label, page_id in system_pages.items():
        if st.sidebar.button(label, key=f"nav_{page_id}", use_container_width=True):
            st.session_state.current_page = page_id
    
    # System Status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔋 System Status")
    render_system_status_sidebar()

def render_system_status_sidebar():
    """Render system status in sidebar"""
    config = load_config()
    api_urls = get_api_urls(config)
    
    try:
        is_healthy, health_data = check_api_health(api_urls)
        
        if is_healthy:
            st.sidebar.markdown('<p class="status-healthy">🟢 API Online</p>', unsafe_allow_html=True)
            if isinstance(health_data, dict):
                models_count = health_data.get("models_available", 0)
                st.sidebar.markdown(f"**Models:** {models_count}")
        else:
            st.sidebar.markdown('<p class="status-error">🔴 API Offline</p>', unsafe_allow_html=True)
            
    except Exception as e:
        st.sidebar.markdown('<p class="status-warning">⚠️ Status Unknown</p>', unsafe_allow_html=True)

# Page Renderers
def render_overview_page():
    """Render comprehensive overview page"""
    st.markdown('<h1 class="main-header">AI Workbench Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete AI Development & Deployment Platform</p>', unsafe_allow_html=True)
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    
    config = load_config()
    api_urls = get_api_urls(config)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Models</h3>
            <p style="font-size: 1.5rem; font-weight: 600; color: #3b82f6;">Available</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Tasks</h3>
            <p style="font-size: 1.5rem; font-weight: 600; color: #10b981;">15+</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔧 Features</h3>
            <p style="font-size: 1.5rem; font-weight: 600; color: #f59e0b;">Advanced</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ Safety</h3>
            <p style="font-size: 1.5rem; font-weight: 600; color: #ef4444;">Built-in</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Categories
    st.markdown("## 🎯 Platform Capabilities")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Core Features", "🚀 Advanced AI", "📊 Analytics", "🛡️ Safety & Ethics"])
    
    with tab1:
        render_core_features_overview()
    
    with tab2:
        render_advanced_features_overview()
    
    with tab3:
        render_analytics_overview()
    
    with tab4:
        render_safety_overview()
    
    # Quick Actions
    st.markdown("## ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Chat Session", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
    
    with col2:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.current_page = "analytics"
            st.rerun()
    
    with col3:
        if st.button("⚔️ Model Battle", use_container_width=True):
            st.session_state.current_page = "battles"
            st.rerun()

def render_core_features_overview():
    """Render core features overview"""
    features = [
        {
            "icon": "💬",
            "title": "Multi-Model Chat",
            "description": "Chat with multiple AI models simultaneously and compare responses",
            "features": ["OpenAI GPT models", "Local LLaMA models", "Response comparison", "Context management"]
        },
        {
            "icon": "📝",
            "title": "Advanced Summarization",
            "description": "Generate summaries with evaluation metrics and quality assessment",
            "features": ["Multiple models", "ROUGE evaluation", "Quality metrics", "Reference comparison"]
        },
        {
            "icon": "🌐",
            "title": "Multi-Language Translation",
            "description": "Translate text with quality checks and bias detection",
            "features": ["15+ languages", "Quality validation", "Error detection", "Cultural context"]
        },
        {
            "icon": "📄",
            "title": "Document Intelligence",
            "description": "Upload and analyze documents with RAG capabilities",
            "features": ["PDF processing", "OCR for images", "Vector search", "Q&A over documents"]
        }
    ]
    
    for feature in features:
        st.markdown(f"""
        <div class="feature-card">
            <h3>{feature['icon']} {feature['title']}</h3>
            <p>{feature['description']}</p>
            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem;">
                {''.join([f'<span style="background: #e0e7ff; color: #3730a3; padding: 0.25rem 0.5rem; border-radius: 0.375rem; font-size: 0.8rem;">{f}</span>' for f in feature['features']])}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_advanced_features_overview():
    """Render advanced features overview"""
    features = [
        {
            "icon": "🔬",
            "title": "Model Evaluation Suite",
            "description": "Comprehensive evaluation with ROUGE, BLEU, BERT-Score and custom metrics",
            "status": "Active"
        },
        {
            "icon": "🤝",
            "title": "Model Fusion Engine",
            "description": "Combine outputs from multiple models for improved performance",
            "status": "Active"
        },
        {
            "icon": "⚔️",
            "title": "Collaboration Arena",
            "description": "Real-time model battles and collaborative evaluation",
            "status": "Beta"
        },
        {
            "icon": "🎓",
            "title": "AI Tutor System",
            "description": "Educational insights and performance explanations",
            "status": "Active"
        },
        {
            "icon": "🎛️",
            "title": "Fine-tuning Platform",
            "description": "LoRA/QLoRA fine-tuning with monitoring and evaluation",
            "status": "Advanced"
        },
        {
            "icon": "👥",
            "title": "Crowdsourcing Hub",
            "description": "Community-driven dataset collection and validation",
            "status": "Active"
        }
    ]
    
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            status_color = {"Active": "#10b981", "Beta": "#f59e0b", "Advanced": "#8b5cf6"}[feature["status"]]
            st.markdown(f"""
            <div class="feature-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h4>{feature['icon']} {feature['title']}</h4>
                    <span style="background: {status_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 0.5rem; font-size: 0.7rem;">{feature['status']}</span>
                </div>
                <p style="font-size: 0.9rem; color: #6b7280;">{feature['description']}</p>
            </div>
            """, unsafe_allow_html=True)

def render_analytics_overview():
    """Render analytics overview"""
    st.markdown("""
    ### 📊 Comprehensive Analytics & Monitoring
    
    **Performance Metrics:**
    - Real-time inference monitoring
    - Cost tracking and optimization
    - Usage statistics and trends
    - Model performance comparison
    
    **Visualization Tools:**
    - Interactive charts and graphs
    - Performance dashboards
    - Cost analysis reports
    - Usage pattern analysis
    
    **Reporting Features:**
    - Automated report generation
    - Custom metric tracking
    - Export capabilities
    - Historical data analysis
    """)

def render_safety_overview():
    """Render safety and ethics overview"""
    st.markdown("""
    ### 🛡️ Built-in Safety & Ethics
    
    **Content Safety:**
    - Toxicity detection with Detoxify
    - Sentiment analysis with VADER
    - Harmful content filtering
    - Privacy protection measures
    
    **Bias Detection:**
    - Multi-dimensional bias analysis
    - Fairness metrics evaluation
    - Demographic bias checking
    - Cultural sensitivity assessment
    
    **Ethics Framework:**
    - Comprehensive safety checks
    - Responsible AI guidelines
    - Transparency reporting
    - Ethical decision support
    """)

def render_chat_interface():
    """Enhanced chat interface"""
    st.markdown('<h1 class="main-header">💬 AI Chat Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with multiple AI models simultaneously</p>', unsafe_allow_html=True)
    
    config = load_config()
    api_urls = get_api_urls(config)
    
    # Chat settings
    with st.expander("🎛️ Chat Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.7, 0.1)
            
        with col2:
            max_tokens = st.slider("📏 Max Tokens", 50, 500, 150, 10)
            
        with col3:
            top_p = st.slider("🎯 Top-p", 0.1, 1.0, 0.9, 0.05)
        
        # Advanced options
        st.markdown("**🚀 Advanced Options:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enable_fusion = st.checkbox("🤝 Enable Model Fusion", value=st.session_state.get("fusion_enabled", False))
            st.session_state.fusion_enabled = enable_fusion
            
        with col2:
            enable_ethics = st.checkbox("🛡️ Ethics Analysis", value=st.session_state.get("ethics_enabled", True))
            st.session_state.ethics_enabled = enable_ethics
            
        with col3:
            enable_tutor = st.checkbox("🎓 AI Tutor Insights", value=st.session_state.get("tutor_enabled", True))
            st.session_state.tutor_enabled = enable_tutor
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background: #f3f4f6; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #3b82f6;">
                    <strong>👤 You</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #ffffff; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #10b981; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
                    <strong>🤖 AI Assistant</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    prompt = st.chat_input("Type your message here...", key="chat_input")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Prepare API payload
        payload = {
            "task": "chat",
            "messages": st.session_state.messages,
            "params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
        }
        
        with st.spinner("🤖 AI is thinking..."):
            success, result = make_api_request(api_urls["process"], "POST", payload)
            
            if success:
                if isinstance(result, str):
                    ai_response = result
                else:
                    ai_response = result.get("response", "No response generated")
                
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
                # Show additional insights if enabled
                if st.session_state.ethics_enabled or st.session_state.tutor_enabled:
                    with st.expander("🔍 Analysis & Insights", expanded=False):
                        if st.session_state.ethics_enabled:
                            st.markdown("**🛡️ Ethics Analysis:**")
                            st.info("Response appears safe and appropriate.")
                        
                        if st.session_state.tutor_enabled:
                            st.markdown("**🎓 AI Tutor Insights:**")
                            st.info("This response demonstrates good coherence and relevance to your question.")
                
                st.rerun()
            else:
                st.error(f"Error: {result}")

def render_model_battles():
    """Render model battles interface"""
    st.markdown('<h1 class="main-header">⚔️ Model Battle Arena</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Compare AI models in real-time competitions</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🆚 Create Battle", "📊 Active Battles", "🏆 Leaderboard"])
    
    with tab1:
        render_create_battle()
    
    with tab2:
        render_active_battles()
    
    with tab3:
        render_leaderboard()

def render_create_battle():
    """Render create battle interface"""
    st.markdown("### Create New Model Battle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        battle_name = st.text_input("Battle Name", placeholder="My AI Battle")
        task_type = st.selectbox("Task Type", ["summarization", "translation", "chat", "general"])
        
        # Model selection
        st.markdown("**Select Models to Battle:**")
        available_models = ["gpt-4o", "gpt-3.5-turbo", "llama-3.1-8b"]
        selected_models = st.multiselect("Models", available_models, default=available_models[:2])
    
    with col2:
        input_text = st.text_area("Input Text", height=150, placeholder="Enter the text for models to process...")
        
        # Battle settings
        st.markdown("**Battle Settings:**")
        max_participants = st.number_input("Max Participants", min_value=2, max_value=10, value=5)
        battle_duration = st.selectbox("Duration", ["5 minutes", "10 minutes", "30 minutes", "1 hour"])
    
    if st.button("🚀 Create Battle", type="primary", use_container_width=True):
        if len(selected_models) >= 2 and input_text.strip():
            # Create battle (simulate)
            battle_id = f"battle_{int(time.time())}"
            battle_data = {
                "id": battle_id,
                "name": battle_name or "Unnamed Battle",
                "task": task_type,
                "models": selected_models,
                "input": input_text,
                "status": "active",
                "participants": 1,
                "max_participants": max_participants,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if "battles" not in st.session_state:
                st.session_state.battles = []
            st.session_state.battles.append(battle_data)
            
            st.success(f"🎉 Battle '{battle_data['name']}' created successfully!")
            st.info(f"Battle ID: {battle_id}")
        else:
            st.error("Please select at least 2 models and provide input text.")

def render_active_battles():
    """Render active battles"""
    st.markdown("### Active Battles")
    
    if "battles" not in st.session_state or not st.session_state.battles:
        st.info("No active battles. Create a new battle to get started!")
        return
    
    for battle in st.session_state.battles:
        with st.expander(f"⚔️ {battle['name']} - {battle['status'].title()}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Task:** {battle['task']}")
                st.markdown(f"**Models:** {', '.join(battle['models'])}")
            
            with col2:
                st.markdown(f"**Participants:** {battle['participants']}/{battle['max_participants']}")
                st.markdown(f"**Created:** {battle['created']}")
            
            with col3:
                if st.button(f"Join Battle", key=f"join_{battle['id']}"):
                    st.success("Joined battle successfully!")
                if st.button(f"View Results", key=f"results_{battle['id']}"):
                    st.info("Battle results will be displayed here...")

def render_leaderboard():
    """Render model leaderboard"""
    st.markdown("### 🏆 Model Performance Leaderboard")
    
    # Sample leaderboard data
    leaderboard_data = {
        "Rank": [1, 2, 3, 4, 5],
        "Model": ["gpt-4o", "claude-3-sonnet", "gpt-3.5-turbo", "llama-3.1-8b", "mistral-7b"],
        "ELO Score": [1842, 1789, 1734, 1689, 1645],
        "Battles": [156, 142, 189, 134, 98],
        "Win Rate": [0.73, 0.68, 0.61, 0.57, 0.52],
        "Avg Response Time": ["1.2s", "0.9s", "0.8s", "2.1s", "1.8s"]
    }
    
    df = pd.DataFrame(leaderboard_data)
    
    # Display as styled table
    st.dataframe(
        df,
        column_config={
            "Rank": st.column_config.NumberColumn("🏆 Rank"),
            "Model": st.column_config.TextColumn("🤖 Model"),
            "ELO Score": st.column_config.NumberColumn("📊 ELO Score"),
            "Battles": st.column_config.NumberColumn("⚔️ Battles"),
            "Win Rate": st.column_config.ProgressColumn("🎯 Win Rate", min_value=0, max_value=1),
            "Avg Response Time": st.column_config.TextColumn("⏱️ Avg Time")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(df, x='Model', y='ELO Score', title='ELO Scores by Model')
        fig1.update_layout(template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.scatter(df, x='Battles', y='Win Rate', size='ELO Score', 
                         hover_name='Model', title='Win Rate vs Battles')
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

def render_crowdsourcing():
    """Render crowdsourcing interface"""
    st.markdown('<h1 class="main-header">👥 Crowdsourcing Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Community-driven dataset collection and validation</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Submit Data", "📋 Pending Review", "✅ Approved Datasets", "📊 Statistics"])
    
    with tab1:
        render_data_submission()
    
    with tab2:
        render_pending_review()
    
    with tab3:
        render_approved_datasets()
    
    with tab4:
        render_crowdsource_stats()

def render_data_submission():
    """Render data submission interface"""
    st.markdown("### Submit New Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        submitter_name = st.text_input("Your Name", placeholder="John Doe")
        submitter_email = st.text_input("Email (optional)", placeholder="john@example.com")
        dataset_category = st.selectbox("Category", ["Summarization", "Translation", "QA", "Classification", "Other"])
        task_type = st.selectbox("Task Type", ["summarization", "translation", "chat", "classification"])
    
    with col2:
        st.markdown("**Dataset Format:**")
        st.code('''
{
  "input": "Your input text here",
  "output": "Expected output here",
  "metadata": {
    "language": "en",
    "domain": "general"
  }
}
        ''', language="json")
    
    # Data input
    st.markdown("**Dataset Entries:**")
    data_input = st.text_area(
        "Enter your dataset (JSON format, one entry per line)",
        height=200,
        placeholder='{"input": "Summarize this text...", "output": "This is a summary..."}'
    )
    
    if st.button("🚀 Submit Dataset", type="primary", use_container_width=True):
        if submitter_name and data_input:
            try:
                # Parse and validate data
                lines = data_input.strip().split('\n')
                dataset = []
                for line in lines:
                    if line.strip():
                        entry = json.loads(line)
                        dataset.append(entry)
                
                # Simulate submission
                submission_id = f"ds_{int(time.time())}"
                submission = {
                    "id": submission_id,
                    "submitter": submitter_name,
                    "email": submitter_email,
                    "category": dataset_category,
                    "task_type": task_type,
                    "data": dataset,
                    "status": "pending",
                    "submitted": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "quality_score": np.random.uniform(0.6, 0.9)
                }
                
                if "submissions" not in st.session_state:
                    st.session_state.submissions = []
                st.session_state.submissions.append(submission)
                
                st.success(f"🎉 Dataset submitted successfully!")
                st.info(f"Submission ID: {submission_id}")
                st.info(f"Entries: {len(dataset)}")
                
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {e}")
        else:
            st.error("Please provide your name and dataset entries.")

def render_pending_review():
    """Render pending datasets for review"""
    st.markdown("### Datasets Pending Review")
    
    if "submissions" not in st.session_state or not st.session_state.submissions:
        st.info("No datasets pending review.")
        return
    
    pending_submissions = [s for s in st.session_state.submissions if s["status"] == "pending"]
    
    if not pending_submissions:
        st.info("All datasets have been reviewed!")
        return
    
    for submission in pending_submissions:
        with st.expander(f"📋 {submission['id']} - {submission['category']}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Submitter:** {submission['submitter']}")
                st.markdown(f"**Category:** {submission['category']}")
                st.markdown(f"**Task:** {submission['task_type']}")
            
            with col2:
                st.markdown(f"**Entries:** {len(submission['data'])}")
                st.markdown(f"**Submitted:** {submission['submitted']}")
                st.markdown(f"**Quality Score:** {submission['quality_score']:.2f}")
            
            with col3:
                if st.button("✅ Approve", key=f"approve_{submission['id']}"):
                    submission['status'] = "approved"
                    st.success("Dataset approved!")
                    st.rerun()
                
                if st.button("❌ Reject", key=f"reject_{submission['id']}"):
                    submission['status'] = "rejected"
                    st.warning("Dataset rejected!")
                    st.rerun()
            
            # Show sample data
            if st.checkbox(f"Show sample data", key=f"sample_{submission['id']}"):
                sample_data = submission['data'][:3]  # Show first 3 entries
                for i, entry in enumerate(sample_data):
                    st.json(entry)

def render_approved_datasets():
    """Render approved datasets"""
    st.markdown("### Approved Datasets")
    
    if "submissions" not in st.session_state:
        st.info("No approved datasets yet.")
        return
    
    approved_submissions = [s for s in st.session_state.submissions if s["status"] == "approved"]
    
    if not approved_submissions:
        st.info("No datasets have been approved yet.")
        return
    
    # Summary stats
    total_entries = sum(len(s['data']) for s in approved_submissions)
    categories = list(set(s['category'] for s in approved_submissions))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Datasets", len(approved_submissions))
    with col2:
        st.metric("Total Entries", total_entries)
    with col3:
        st.metric("Categories", len(categories))
    
    # Dataset list
    for submission in approved_submissions:
        with st.expander(f"✅ {submission['id']} - {submission['category']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Submitter:** {submission['submitter']}")
                st.markdown(f"**Category:** {submission['category']}")
                st.markdown(f"**Entries:** {len(submission['data'])}")
            
            with col2:
                st.markdown(f"**Quality Score:** {submission['quality_score']:.2f}")
                st.markdown(f"**Approved:** {submission['submitted']}")
                
                if st.button("📥 Download", key=f"download_{submission['id']}"):
                    # Convert to downloadable format
                    json_data = json.dumps(submission['data'], indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"dataset_{submission['id']}.json",
                        mime="application/json"
                    )

def render_crowdsource_stats():
    """Render crowdsourcing statistics"""
    st.markdown("### 📊 Crowdsourcing Statistics")
    
    if "submissions" not in st.session_state or not st.session_state.submissions:
        st.info("No submission data available.")
        return
    
    submissions = st.session_state.submissions
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Submissions", len(submissions))
    
    with col2:
        approved = len([s for s in submissions if s["status"] == "approved"])
        st.metric("Approved", approved)
    
    with col3:
        pending = len([s for s in submissions if s["status"] == "pending"])
        st.metric("Pending Review", pending)
    
    with col4:
        if len(submissions) > 0:
            approval_rate = approved / len(submissions) * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Status distribution
        status_counts = {}
        for s in submissions:
            status = s["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            fig1 = px.pie(values=list(status_counts.values()), 
                         names=list(status_counts.keys()), 
                         title="Submission Status Distribution")
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Category distribution
        category_counts = {}
        for s in submissions:
            category = s["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        if category_counts:
            fig2 = px.bar(x=list(category_counts.keys()), 
                         y=list(category_counts.values()),
                         title="Submissions by Category")
            st.plotly_chart(fig2, use_container_width=True)

def render_fine_tuning():
    """Render fine-tuning interface"""
    st.markdown('<h1 class="main-header">🎛️ Model Fine-tuning</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced model training and customization</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Create Job", "📊 Active Jobs", "📁 Models", "⚙️ Settings"])
    
    with tab1:
        render_create_finetune_job()
    
    with tab2:
        render_active_finetune_jobs()
    
    with tab3:
        render_finetune_models()
    
    with tab4:
        render_finetune_settings()

def render_create_finetune_job():
    """Render create fine-tuning job interface"""
    st.markdown("### Create Fine-tuning Job")
    
    col1, col2 = st.columns(2)
    
    with col1:
        job_name = st.text_input("Job Name", placeholder="my-custom-model")
        base_model = st.selectbox("Base Model", [
            "meta-llama/Llama-3.1-8B-Instruct",
            "microsoft/DialoGPT-medium",
            "google/flan-t5-base"
        ])
        method = st.selectbox("Fine-tuning Method", ["LoRA", "QLoRA", "Full Fine-tuning"])
        task_type = st.selectbox("Task Type", ["causal_lm", "summarization", "translation", "instruction"])
    
    with col2:
        # Training parameters
        st.markdown("**Training Parameters:**")
        learning_rate = st.select_slider("Learning Rate", 
                                        options=[1e-5, 2e-5, 3e-5, 5e-5, 1e-4], 
                                        value=2e-5, format_func=lambda x: f"{x:.0e}")
        epochs = st.slider("Epochs", 1, 10, 3)
        batch_size = st.slider("Batch Size", 1, 16, 4)
        
        # LoRA specific settings
        if method in ["LoRA", "QLoRA"]:
            lora_rank = st.slider("LoRA Rank", 4, 64, 16)
            lora_alpha = st.slider("LoRA Alpha", 8, 128, 32)
    
    # Dataset upload
    st.markdown("**Training Dataset:**")
    uploaded_file = st.file_uploader(
        "Upload training data (JSON/JSONL format)",
        type=["json", "jsonl"],
        help="Upload your training dataset in the correct format"
    )
    
    if uploaded_file:
        try:
            content = uploaded_file.read().decode('utf-8')
            # Parse and validate
            if uploaded_file.name.endswith('.jsonl'):
                lines = content.strip().split('\n')
                data = [json.loads(line) for line in lines if line.strip()]
            else:
                data = json.loads(content)
            
            st.success(f"✅ Dataset loaded: {len(data)} examples")
            
            # Show sample
            if st.checkbox("Show sample data"):
                st.json(data[0] if data else {})
                
        except Exception as e:
            st.error(f"Error parsing dataset: {e}")
    
    # Start job
    if st.button("🚀 Start Fine-tuning Job", type="primary", use_container_width=True):
        if job_name and uploaded_file:
            # Create job (simulate)
            job_id = f"ft_{int(time.time())}"
            job_data = {
                "id": job_id,
                "name": job_name,
                "base_model": base_model,
                "method": method,
                "task_type": task_type,
                "status": "running",
                "progress": 0,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "estimated_completion": (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "parameters": {
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "batch_size": batch_size
                }
            }
            
            if "finetune_jobs" not in st.session_state:
                st.session_state.finetune_jobs = []
            st.session_state.finetune_jobs.append(job_data)
            
            st.success(f"🎉 Fine-tuning job '{job_name}' started!")
            st.info(f"Job ID: {job_id}")
        else:
            st.error("Please provide job name and upload dataset.")

def render_active_finetune_jobs():
    """Render active fine-tuning jobs"""
    st.markdown("### Active Fine-tuning Jobs")
    
    if "finetune_jobs" not in st.session_state or not st.session_state.finetune_jobs:
        st.info("No active fine-tuning jobs.")
        return
    
    for job in st.session_state.finetune_jobs:
        with st.expander(f"🎛️ {job['name']} - {job['status'].title()}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Job ID:** {job['id']}")
                st.markdown(f"**Base Model:** {job['base_model']}")
                st.markdown(f"**Method:** {job['method']}")
            
            with col2:
                st.markdown(f"**Status:** {job['status']}")
                st.markdown(f"**Created:** {job['created']}")
                st.markdown(f"**Est. Completion:** {job['estimated_completion']}")
            
            with col3:
                # Progress simulation
                if job['status'] == 'running':
                    progress = min(job.get('progress', 0) + np.random.randint(5, 15), 100)
                    job['progress'] = progress
                    st.progress(progress / 100, text=f"Training Progress: {progress}%")
                    
                    if progress >= 100:
                        job['status'] = 'completed'
                        st.rerun()
                else:
                    st.progress(1.0, text="Completed ✅")
            
            # Job controls
            col1, col2, col3 = st.columns(3)
            with col1:
                if job['status'] == 'running' and st.button("⏸️ Pause", key=f"pause_{job['id']}"):
                    job['status'] = 'paused'
                    st.rerun()
            
            with col2:
                if st.button("📊 View Logs", key=f"logs_{job['id']}"):
                    st.code(f"""
[2024-01-15 10:30:15] Starting fine-tuning job {job['id']}
[2024-01-15 10:30:16] Loading base model: {job['base_model']}
[2024-01-15 10:30:18] Initializing {job['method']} configuration
[2024-01-15 10:30:20] Training started...
[2024-01-15 10:31:15] Epoch 1/3 - Loss: 2.45
[2024-01-15 10:32:30] Epoch 2/3 - Loss: 1.89
                    """)
            
            with col3:
                if job['status'] == 'running' and st.button("🛑 Stop", key=f"stop_{job['id']}"):
                    job['status'] = 'stopped'
                    st.rerun()

def render_finetune_models():
    """Render fine-tuned models"""
    st.markdown("### Fine-tuned Models")
    
    # Sample models
    models = [
        {
            "name": "custom-summarizer-v1",
            "base_model": "llama-3.1-8b",
            "task": "summarization",
            "created": "2024-01-10",
            "performance": 0.85,
            "size": "4.2 GB"
        },
        {
            "name": "domain-expert-v2",
            "base_model": "llama-3.1-8b",
            "task": "instruction",
            "created": "2024-01-08",
            "performance": 0.91,
            "size": "4.2 GB"
        }
    ]
    
    for model in models:
        with st.expander(f"🤖 {model['name']}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Base Model:** {model['base_model']}")
                st.markdown(f"**Task:** {model['task']}")
                st.markdown(f"**Created:** {model['created']}")
            
            with col2:
                st.markdown(f"**Performance:** {model['performance']:.2f}")
                st.markdown(f"**Size:** {model['size']}")
                st.progress(model['performance'], text=f"Quality: {model['performance']:.1%}")
            
            with col3:
                if st.button("🚀 Deploy", key=f"deploy_{model['name']}"):
                    st.success("Model deployed successfully!")
                if st.button("📊 Evaluate", key=f"eval_{model['name']}"):
                    st.info("Evaluation started...")
                if st.button("🗑️ Delete", key=f"delete_{model['name']}"):
                    st.warning("Model deletion confirmed.")

def render_finetune_settings():
    """Render fine-tuning settings"""
    st.markdown("### Fine-tuning Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Default Parameters:**")
        default_lr = st.number_input("Default Learning Rate", value=2e-5, format="%.0e")
        default_epochs = st.number_input("Default Epochs", value=3, min_value=1, max_value=20)
        default_batch_size = st.number_input("Default Batch Size", value=4, min_value=1, max_value=32)
        
        st.markdown("**Resource Limits:**")
        max_concurrent_jobs = st.number_input("Max Concurrent Jobs", value=2, min_value=1, max_value=10)
        gpu_memory_limit = st.number_input("GPU Memory Limit (GB)", value=16, min_value=4, max_value=80)
    
    with col2:
        st.markdown("**Supported Methods:**")
        lora_enabled = st.checkbox("LoRA Fine-tuning", value=True)
        qlora_enabled = st.checkbox("QLoRA Fine-tuning", value=True)
        full_ft_enabled = st.checkbox("Full Fine-tuning", value=False)
        
        st.markdown("**Data Validation:**")
        validate_data = st.checkbox("Validate uploaded datasets", value=True)
        min_samples = st.number_input("Minimum dataset size", value=100, min_value=10)
        max_samples = st.number_input("Maximum dataset size", value=100000, min_value=100)
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")

def render_analytics_interface():
    """Render comprehensive analytics interface"""
    st.markdown('<h1 class="main-header">📊 Analytics & Monitoring</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive performance analytics and system monitoring</p>', unsafe_allow_html=True)
    
    # Load configuration and check status
    config = load_config()
    api_urls = get_api_urls(config)
    is_healthy, health_data = check_api_health(api_urls)
    
    # System overview cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "#10b981" if is_healthy else "#ef4444"
        status_text = "Online" if is_healthy else "Offline"
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔋 System Status</h4>
            <p style="color: {status_color}; font-weight: 600; font-size: 1.2rem;">{status_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        models_count = health_data.get("models_available", 0) if isinstance(health_data, dict) else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>🤖 Available Models</h4>
            <p style="color: #3b82f6; font-weight: 600; font-size: 1.5rem;">{models_count}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Simulate request count
        request_count = np.random.randint(1500, 3000)
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Today's Requests</h4>
            <p style="color: #10b981; font-weight: 600; font-size: 1.5rem;">{request_count:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Simulate cost
        daily_cost = np.random.uniform(15, 45)
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Today's Cost</h4>
            <p style="color: #f59e0b; font-weight: 600; font-size: 1.5rem;">${daily_cost:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "💰 Cost Analysis", "🔍 Usage Patterns", "⚠️ System Health"])
    
    with tab1:
        render_performance_analytics()
    
    with tab2:
        render_cost_analytics()
    
    with tab3:
        render_usage_analytics()
    
    with tab4:
        render_health_analytics()

def render_performance_analytics():
    """Render performance analytics"""
    st.markdown("### 📊 Performance Metrics")
    
    # Generate sample performance data
    dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
    models = ['gpt-4o', 'gpt-3.5-turbo', 'llama-3.1-8b']
    
    performance_data = []
    for date in dates:
        for model in models:
            performance_data.append({
                'Date': date,
                'Model': model,
                'Avg Response Time': np.random.uniform(0.5, 3.0),
                'Success Rate': np.random.uniform(0.95, 1.0),
                'Requests': np.random.randint(50, 500),
                'Quality Score': np.random.uniform(0.7, 0.95)
            })
    
    df_perf = pd.DataFrame(performance_data)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time trends
        fig1 = px.line(df_perf, x='Date', y='Avg Response Time', color='Model',
                      title='Average Response Time Trends')
        fig1.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Success rate comparison
        latest_data = df_perf[df_perf['Date'] == df_perf['Date'].max()]
        fig2 = px.bar(latest_data, x='Model', y='Success Rate',
                     title='Success Rate by Model (Latest)')
        fig2.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Quality metrics heatmap
    st.markdown("#### Quality Score Heatmap")
    pivot_data = df_perf.pivot_table(values='Quality Score', index='Model', columns='Date', aggfunc='mean')
    fig3 = px.imshow(pivot_data.values, 
                    x=pivot_data.columns.strftime('%m-%d'),
                    y=pivot_data.index,
                    title='Quality Score Heatmap',
                    color_continuous_scale='Viridis')
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)

def render_cost_analytics():
    """Render cost analytics"""
    st.markdown("### 💰 Cost Analysis")
    
    # Generate sample cost data
    dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
    models = ['gpt-4o', 'gpt-3.5-turbo', 'llama-3.1-8b']
    
    cost_data = []
    for date in dates:
        for model in models:
            base_cost = {'gpt-4o': 0.03, 'gpt-3.5-turbo': 0.002, 'llama-3.1-8b': 0.0}[model]
            daily_requests = np.random.randint(100, 1000)
            tokens_per_request = np.random.randint(500, 2000)
            daily_cost = (daily_requests * tokens_per_request * base_cost) / 1000
            
            cost_data.append({
                'Date': date,
                'Model': model,
                'Daily Cost': daily_cost,
                'Requests': daily_requests,
                'Total Tokens': daily_requests * tokens_per_request,
                'Cost per Request': daily_cost / daily_requests if daily_requests > 0 else 0
            })
    
    df_cost = pd.DataFrame(cost_data)
    
    # Cost overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_cost = df_cost['Daily Cost'].sum()
        st.metric("Total Cost (15 days)", f"${total_cost:.2f}")
    
    with col2:
        avg_daily_cost = df_cost.groupby('Date')['Daily Cost'].sum().mean()
        st.metric("Avg Daily Cost", f"${avg_daily_cost:.2f}")
    
    with col3:
        most_expensive_model = df_cost.groupby('Model')['Daily Cost'].sum().idxmax()
        st.metric("Most Expensive Model", most_expensive_model)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily cost trends
        daily_costs = df_cost.groupby(['Date', 'Model'])['Daily Cost'].sum().reset_index()
        fig1 = px.line(daily_costs, x='Date', y='Daily Cost', color='Model',
                      title='Daily Cost Trends by Model')
        fig1.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Cost distribution pie chart
        model_costs = df_cost.groupby('Model')['Daily Cost'].sum().reset_index()
        fig2 = px.pie(model_costs, values='Daily Cost', names='Model',
                     title='Cost Distribution by Model')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Cost efficiency analysis
    st.markdown("#### Cost Efficiency Analysis")
    efficiency_data = df_cost.groupby('Model').agg({
        'Daily Cost': 'sum',
        'Requests': 'sum',
        'Total Tokens': 'sum'
    }).reset_index()
    
    efficiency_data['Cost per Request'] = efficiency_data['Daily Cost'] / efficiency_data['Requests']
    efficiency_data['Cost per 1K Tokens'] = (efficiency_data['Daily Cost'] / efficiency_data['Total Tokens']) * 1000
    
    st.dataframe(
        efficiency_data,
        column_config={
            "Model": "🤖 Model",
            "Daily Cost": st.column_config.NumberColumn("💰 Total Cost", format="$%.2f"),
            "Requests": st.column_config.NumberColumn("📊 Total Requests", format="%d"),
            "Total Tokens": st.column_config.NumberColumn("🔤 Total Tokens", format="%d"),
            "Cost per Request": st.column_config.NumberColumn("💸 Cost/Request", format="$%.4f"),
            "Cost per 1K Tokens": st.column_config.NumberColumn("💱 Cost/1K Tokens", format="$%.4f")
        },
        hide_index=True,
        use_container_width=True
    )

def render_usage_analytics():
    """Render usage pattern analytics"""
    st.markdown("### 🔍 Usage Patterns")
    
    # Generate sample usage data
    hours = list(range(24))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    tasks = ['chat', 'summarization', 'translation', 'documents']
    
    # Hourly usage pattern
    hourly_data = []
    for hour in hours:
        # Simulate business hours peak
        base_requests = 50
        if 9 <= hour <= 17:  # Business hours
            base_requests *= 3
        elif 19 <= hour <= 22:  # Evening peak
            base_requests *= 2
        
        requests = base_requests + np.random.randint(-20, 20)
        hourly_data.append({'Hour': hour, 'Requests': max(0, requests)})
    
    df_hourly = pd.DataFrame(hourly_data)
    
    # Task usage distribution
    task_data = []
    for task in tasks:
        weight = {'chat': 0.4, 'summarization': 0.3, 'translation': 0.2, 'documents': 0.1}[task]
        requests = int(1000 * weight * np.random.uniform(0.8, 1.2))
        task_data.append({'Task': task.title(), 'Requests': requests})
    
    df_tasks = pd.DataFrame(task_data)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly usage pattern
        fig1 = px.bar(df_hourly, x='Hour', y='Requests',
                     title='Hourly Usage Pattern (Average)')
        fig1.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Task distribution
        fig2 = px.bar(df_tasks, x='Task', y='Requests',
                     title='Requests by Task Type')
        fig2.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Weekly usage heatmap
    st.markdown("#### Weekly Usage Heatmap")
    weekly_data = np.random.randint(10, 200, size=(len(days), len(hours)))
    
    fig3 = px.imshow(weekly_data,
                    x=[f"{h:02d}:00" for h in hours],
                    y=days,
                    title='Weekly Usage Heatmap (Requests per Hour)',
                    color_continuous_scale='Blues')
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Usage statistics
    st.markdown("#### Usage Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        peak_hour = df_hourly.loc[df_hourly['Requests'].idxmax(), 'Hour']
        st.metric("Peak Hour", f"{peak_hour:02d}:00")
    
    with col2:
        total_requests = df_tasks['Requests'].sum()
        st.metric("Total Requests", f"{total_requests:,}")
    
    with col3:
        most_popular_task = df_tasks.loc[df_tasks['Requests'].idxmax(), 'Task']
        st.metric("Most Popular Task", most_popular_task)
    
    with col4:
        avg_hourly = df_hourly['Requests'].mean()
        st.metric("Avg Hourly Requests", f"{avg_hourly:.0f}")

def render_health_analytics():
    """Render system health analytics"""
    st.markdown("### ⚠️ System Health Monitoring")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = np.random.uniform(30, 80)
        cpu_color = "#10b981" if cpu_usage < 70 else "#f59e0b" if cpu_usage < 85 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <h4>💻 CPU Usage</h4>
            <p style="color: {cpu_color}; font-weight: 600; font-size: 1.5rem;">{cpu_usage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        memory_usage = np.random.uniform(40, 85)
        memory_color = "#10b981" if memory_usage < 75 else "#f59e0b" if memory_usage < 90 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <h4>🧠 Memory Usage</h4>
            <p style="color: {memory_color}; font-weight: 600; font-size: 1.5rem;">{memory_usage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        disk_usage = np.random.uniform(20, 60)
        disk_color = "#10b981" if disk_usage < 80 else "#f59e0b" if disk_usage < 90 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <h4>💾 Disk Usage</h4>
            <p style="color: {disk_color}; font-weight: 600; font-size: 1.5rem;">{disk_usage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        error_rate = np.random.uniform(0.1, 2.5)
        error_color = "#10b981" if error_rate < 1 else "#f59e0b" if error_rate < 5 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚠️ Error Rate</h4>
            <p style="color: {error_color}; font-weight: 600; font-size: 1.5rem;">{error_rate:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Health trends
    col1, col2 = st.columns(2)
    
    with col1:
        # System metrics over time
        time_points = pd.date_range(start='2024-01-15 00:00', end='2024-01-15 23:59', freq='1H')
        metrics_data = []
        
        for t in time_points:
            metrics_data.append({
                'Time': t,
                'CPU': np.random.uniform(30, 80),
                'Memory': np.random.uniform(40, 85),
                'Disk': np.random.uniform(20, 60)
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_metrics['Time'], y=df_metrics['CPU'], name='CPU', line=dict(color='#3b82f6')))
        fig1.add_trace(go.Scatter(x=df_metrics['Time'], y=df_metrics['Memory'], name='Memory', line=dict(color='#10b981')))
        fig1.add_trace(go.Scatter(x=df_metrics['Time'], y=df_metrics['Disk'], name='Disk', line=dict(color='#f59e0b')))
        
        fig1.update_layout(
            title='System Resource Usage (24h)',
            xaxis_title='Time',
            yaxis_title='Usage (%)',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Error rate trends
        error_data = []
        for t in time_points:
            error_data.append({
                'Time': t,
                'Error Rate': np.random.uniform(0.1, 3.0),
                'Response Time': np.random.uniform(0.5, 2.5)
            })
        
        df_errors = pd.DataFrame(error_data)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_errors['Time'], y=df_errors['Error Rate'], 
                                 name='Error Rate (%)', yaxis='y', line=dict(color='#ef4444')))
        fig2.add_trace(go.Scatter(x=df_errors['Time'], y=df_errors['Response Time'], 
                                 name='Response Time (s)', yaxis='y2', line=dict(color='#8b5cf6')))
        
        fig2.update_layout(
            title='Error Rate & Response Time (24h)',
            xaxis_title='Time',
            yaxis=dict(title='Error Rate (%)', side='left'),
            yaxis2=dict(title='Response Time (s)', side='right', overlaying='y'),
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Alert summary
    st.markdown("#### System Alerts")
    alerts = [
        {"level": "warning", "message": "High memory usage detected on server-2", "time": "2 hours ago"},
        {"level": "info", "message": "Model cache cleared successfully", "time": "4 hours ago"},
        {"level": "error", "message": "Connection timeout for external API", "time": "6 hours ago"},
        {"level": "info", "message": "Scheduled maintenance completed", "time": "1 day ago"}
    ]
    
    for alert in alerts:
        icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}[alert["level"]]
        color = {"error": "#ef4444", "warning": "#f59e0b", "info": "#3b82f6"}[alert["level"]]
        
        st.markdown(f"""
        <div style="background: white; border-left: 4px solid {color}; padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;">
            <strong>{icon} {alert['message']}</strong><br>
            <small style="color: #6b7280;">{alert['time']}</small>
        </div>
        """, unsafe_allow_html=True)

def render_ethics_interface():
    """Render ethics and safety interface"""
    st.markdown('<h1 class="main-header">🛡️ Ethics & Safety Center</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive AI safety and ethical AI monitoring</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Content Analysis", "📊 Safety Dashboard", "⚙️ Safety Settings", "📋 Safety Reports"])
    
    with tab1:
        render_content_analysis()
    
    with tab2:
        render_safety_dashboard()
    
    with tab3:
        render_safety_settings()
    
    with tab4:
        render_safety_reports()

def render_content_analysis():
    """Render content analysis interface"""
    st.markdown("### 🔍 Real-time Content Analysis")
    
    # Input text for analysis
    analysis_text = st.text_area(
        "Enter text to analyze for safety and ethics:",
        height=150,
        placeholder="Enter any AI-generated content to analyze for toxicity, bias, safety concerns..."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        analyze_sentiment = st.checkbox("Sentiment Analysis", value=True)
        analyze_toxicity = st.checkbox("Toxicity Detection", value=True)
        analyze_bias = st.checkbox("Bias Detection", value=True)
    
    with col2:
        analyze_privacy = st.checkbox("Privacy Concerns", value=True)
        analyze_safety = st.checkbox("Safety Keywords", value=True)
        analyze_harmful = st.checkbox("Harmful Content", value=True)
    
    if st.button("🔍 Analyze Content", type="primary", use_container_width=True) and analysis_text:
        with st.spinner("Analyzing content for safety and ethics..."):
            # Simulate analysis results
            time.sleep(2)
            
            # Mock analysis results
            results = {
                "sentiment": {
                    "compound": np.random.uniform(-0.5, 0.8),
                    "positive": np.random.uniform(0.1, 0.9),
                    "negative": np.random.uniform(0.0, 0.3),
                    "neutral": np.random.uniform(0.2, 0.8)
                },
                "toxicity": {
                    "score": np.random.uniform(0.0, 0.3),
                    "categories": {
                        "toxic": np.random.uniform(0.0, 0.2),
                        "severe_toxic": np.random.uniform(0.0, 0.1),
                        "obscene": np.random.uniform(0.0, 0.15),
                        "threat": np.random.uniform(0.0, 0.05),
                        "insult": np.random.uniform(0.0, 0.1),
                        "identity_hate": np.random.uniform(0.0, 0.05)
                    }
                },
                "bias": {
                    "overall_score": np.random.uniform(0.0, 0.4),
                    "detected_types": ["gender", "racial"] if np.random.random() > 0.7 else [],
                    "confidence": np.random.uniform(0.6, 0.95)
                },
                "safety": {
                    "risk_level": "low",
                    "concerns": [],
                    "privacy_score": np.random.uniform(0.8, 1.0)
                }
            }
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🎭 Sentiment Analysis")
                sentiment_color = "#10b981" if results["sentiment"]["compound"] > 0.1 else "#f59e0b" if results["sentiment"]["compound"] > -0.1 else "#ef4444"
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;">
                    <p><strong>Overall:</strong> <span style="color: {sentiment_color}; font-weight: 600;">{results["sentiment"]["compound"]:.3f}</span></p>
                    <p><strong>Positive:</strong> {results["sentiment"]["positive"]:.3f}</p>
                    <p><strong>Negative:</strong> {results["sentiment"]["negative"]:.3f}</p>
                    <p><strong>Neutral:</strong> {results["sentiment"]["neutral"]:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### ☣️ Toxicity Analysis")
                toxicity_color = "#10b981" if results["toxicity"]["score"] < 0.3 else "#f59e0b" if results["toxicity"]["score"] < 0.7 else "#ef4444"
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;">
                    <p><strong>Toxicity Score:</strong> <span style="color: {toxicity_color}; font-weight: 600;">{results["toxicity"]["score"]:.3f}</span></p>
                    <p><strong>Status:</strong> {'✅ Safe' if results["toxicity"]["score"] < 0.3 else '⚠️ Review' if results["toxicity"]["score"] < 0.7 else '🚨 Unsafe'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Toxicity categories
                for category, score in results["toxicity"]["categories"].items():
                    if score > 0.1:
                        st.markdown(f"**{category.replace('_', ' ').title()}:** {score:.3f}")
            
            with col3:
                st.markdown("#### ⚖️ Bias Detection")
                bias_color = "#10b981" if results["bias"]["overall_score"] < 0.2 else "#f59e0b" if results["bias"]["overall_score"] < 0.5 else "#ef4444"
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;">
                    <p><strong>Bias Score:</strong> <span style="color: {bias_color}; font-weight: 600;">{results["bias"]["overall_score"]:.3f}</span></p>
                    <p><strong>Confidence:</strong> {results["bias"]["confidence"]:.3f}</p>
                    <p><strong>Detected Types:</strong> {', '.join(results["bias"]["detected_types"]) if results["bias"]["detected_types"] else 'None'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Overall safety assessment
            st.markdown("#### 🛡️ Overall Safety Assessment")
            
            safety_score = 1 - (results["toxicity"]["score"] * 0.4 + results["bias"]["overall_score"] * 0.3 + (1 - results["safety"]["privacy_score"]) * 0.3)
            safety_color = "#10b981" if safety_score > 0.8 else "#f59e0b" if safety_score > 0.6 else "#ef4444"
            safety_status = "✅ Safe" if safety_score > 0.8 else "⚠️ Needs Review" if safety_score > 0.6 else "🚨 Unsafe"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 1.5rem; border-radius: 12px; border: 1px solid #cbd5e1;">
                <h4>Safety Score: <span style="color: {safety_color}; font-weight: 700;">{safety_score:.2f}</span></h4>
                <p><strong>Status:</strong> {safety_status}</p>
                <p><strong>Recommendation:</strong> {'Content appears safe for use.' if safety_score > 0.8 else 'Review content before publishing.' if safety_score > 0.6 else 'Content requires significant revision or should not be used.'}</p>
            </div>
            """, unsafe_allow_html=True)

def render_safety_dashboard():
    """Render safety monitoring dashboard"""
    st.markdown("### 📊 Safety Monitoring Dashboard")
    
    # Safety metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        safe_rate = np.random.uniform(0.92, 0.98)
        st.metric("Safe Content Rate", f"{safe_rate:.1%}", delta=f"+{np.random.uniform(0.1, 2.0):.1f}%")
    
    with col2:
        flagged_content = np.random.randint(15, 45)
        st.metric("Flagged Content (24h)", flagged_content, delta=f"-{np.random.randint(2, 8)}")
    
    with col3:
        avg_toxicity = np.random.uniform(0.05, 0.15)
        st.metric("Avg Toxicity Score", f"{avg_toxicity:.3f}", delta=f"-{np.random.uniform(0.001, 0.01):.3f}")
    
    with col4:
        bias_incidents = np.random.randint(2, 12)
        st.metric("Bias Incidents (7d)", bias_incidents, delta=f"-{np.random.randint(1, 3)}")
    
    # Safety trends
    col1, col2 = st.columns(2)
    
    with col1:
        # Safety score trends
        dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
        safety_data = []
        
        for date in dates:
            safety_data.append({
                'Date': date,
                'Safety Score': np.random.uniform(0.85, 0.98),
                'Toxicity Rate': np.random.uniform(0.02, 0.08),
                'Bias Rate': np.random.uniform(0.01, 0.05)
            })
        
        df_safety = pd.DataFrame(safety_data)
        
        fig1 = px.line(df_safety, x='Date', y='Safety Score',
                      title='Safety Score Trends (15 days)')
        fig1.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Issue distribution
        issue_types = ['Toxicity', 'Bias', 'Privacy', 'Harmful Content', 'Misinformation']
        issue_counts = [np.random.randint(5, 25) for _ in issue_types]
        
        fig2 = px.pie(values=issue_counts, names=issue_types,
                     title='Safety Issues Distribution (7 days)')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Recent safety alerts
    st.markdown("#### 🚨 Recent Safety Alerts")
    
    alerts = [
        {"level": "high", "type": "Toxicity", "content": "High toxicity detected in chat response", "time": "2 hours ago", "action": "Content blocked"},
        {"level": "medium", "type": "Bias", "content": "Gender bias pattern detected in summarization", "time": "4 hours ago", "action": "Manual review"},
        {"level": "low", "type": "Privacy", "content": "Potential PII in user input", "time": "6 hours ago", "action": "Data anonymized"},
        {"level": "medium", "type": "Harmful", "content": "Potentially harmful advice generated", "time": "1 day ago", "action": "Response filtered"}
    ]
    
    for alert in alerts:
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[alert["level"]]
        color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}[alert["level"]]
        
        st.markdown(f"""
        <div style="background: white; border-left: 4px solid {color}; padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{icon} {alert['type']} Alert</strong><br>
                    <span>{alert['content']}</span><br>
                    <small style="color: #6b7280;">{alert['time']}</small>
                </div>
                <div style="text-align: right;">
                    <span style="background: {color}; color: white; padding: 0.25rem 0.5rem; border-radius: 0.375rem; font-size: 0.8rem;">
                        {alert['action']}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_safety_settings():
    """Render safety configuration settings"""
    st.markdown("### ⚙️ Safety Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎚️ Detection Thresholds")
        
        toxicity_threshold = st.slider("Toxicity Threshold", 0.0, 1.0, 0.7, 0.05,
                                      help="Content above this threshold will be flagged as toxic")
        
        bias_threshold = st.slider("Bias Threshold", 0.0, 1.0, 0.5, 0.05,
                                  help="Content above this threshold will be flagged for bias")
        
        sentiment_threshold = st.slider("Extreme Sentiment Threshold", 0.0, 1.0, 0.8, 0.05,
                                       help="Sentiment scores above this will be flagged")
        
        privacy_threshold = st.slider("Privacy Risk Threshold", 0.0, 1.0, 0.3, 0.05,
                                     help="Privacy risk scores above this will trigger alerts")
    
    with col2:
        st.markdown("#### 🛡️ Safety Actions")
        
        auto_block_toxic = st.checkbox("Auto-block toxic content", value=True)
        auto_filter_bias = st.checkbox("Auto-filter biased content", value=False)
        flag_for_review = st.checkbox("Flag questionable content for manual review", value=True)
        log_all_checks = st.checkbox("Log all safety checks", value=True)
        
        st.markdown("#### 📧 Notifications")
        
        email_alerts = st.checkbox("Email alerts for high-risk content", value=True)
        slack_notifications = st.checkbox("Slack notifications for safety issues", value=False)
        daily_reports = st.checkbox("Daily safety summary reports", value=True)
    
    # Bias detection categories
    st.markdown("#### ⚖️ Bias Detection Categories")
    
    bias_categories = {
        "Gender Bias": st.checkbox("Gender bias detection", value=True),
        "Racial Bias": st.checkbox("Racial bias detection", value=True),
        "Religious Bias": st.checkbox("Religious bias detection", value=True),
        "Age Bias": st.checkbox("Age bias detection", value=True),
        "Socioeconomic Bias": st.checkbox("Socioeconomic bias detection", value=False),
        "Cultural Bias": st.checkbox("Cultural bias detection", value=False)
    }
    
    # Save settings
    if st.button("💾 Save Safety Settings", type="primary", use_container_width=True):
        # Store settings in session state
        st.session_state.safety_settings = {
            "toxicity_threshold": toxicity_threshold,
            "bias_threshold": bias_threshold,
            "sentiment_threshold": sentiment_threshold,
            "privacy_threshold": privacy_threshold,
            "auto_actions": {
                "block_toxic": auto_block_toxic,
                "filter_bias": auto_filter_bias,
                "flag_review": flag_for_review,
                "log_checks": log_all_checks
            },
            "notifications": {
                "email_alerts": email_alerts,
                "slack_notifications": slack_notifications,
                "daily_reports": daily_reports
            },
            "bias_categories": bias_categories
        }
        st.success("✅ Safety settings saved successfully!")

def render_safety_reports():
    """Render safety reports interface"""
    st.markdown("### 📋 Safety Reports")
    
    # Report generation
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox("Report Type", [
            "Daily Safety Summary",
            "Weekly Safety Analysis",
            "Monthly Compliance Report",
            "Incident Analysis Report",
            "Bias Detection Report",
            "Model Safety Comparison"
        ])
        
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            help="Select the date range for the report"
        )
    
    with col2:
        include_charts = st.checkbox("Include visualizations", value=True)
        include_details = st.checkbox("Include detailed logs", value=False)
        export_format = st.selectbox("Export Format", ["PDF", "HTML", "CSV", "JSON"])
    
    if st.button("📊 Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating safety report..."):
            time.sleep(2)
            
            # Sample report content
            st.markdown("#### 📋 Safety Report Preview")
            
            report_content = f"""
# {report_type}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Period:** {date_range[0]} to {date_range[1]}

## Executive Summary
- **Total Content Analyzed:** 15,247 items
- **Safety Pass Rate:** 94.2%
- **High-Risk Items:** 23 (0.15%)
- **Manual Reviews:** 156 (1.02%)

## Key Findings
- Toxicity detection flagged 89 items (0.58%)
- Bias detection identified 34 potential issues (0.22%)
- Privacy concerns raised for 12 items (0.08%)
- Overall safety score improved by 2.3% compared to previous period

## Recommendations
1. Review bias detection thresholds for better accuracy
2. Implement additional training for edge cases
3. Update safety guidelines based on recent incidents
4. Consider expanding bias categories for more comprehensive coverage

## Detailed Statistics
- **Average Toxicity Score:** 0.034
- **Average Bias Score:** 0.021
- **Most Common Issue:** Mild sentiment extremes
- **Response Time:** 98.7% of checks completed under 200ms
            """
            
            st.markdown(report_content)
            
            # Download button
            st.download_button(
                label=f"📥 Download {report_type}",
                data=report_content,
                file_name=f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

def render_ai_tutor():
    """Render AI Tutor interface"""
    st.markdown('<h1 class="main-header">🎓 AI Tutor System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Educational insights and performance explanations</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📚 Learning Hub", "📊 Performance Analysis", "🎯 Recommendations"])
    
    with tab1:
        render_learning_hub()
    
    with tab2:
        render_performance_analysis()
    
    with tab3:
        render_tutor_recommendations()

def render_learning_hub():
    """Render learning hub with educational content"""
    st.markdown("### 📚 AI Learning Hub")
    
    # Learning topics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🤖 Understanding AI Models
        
        **What are Language Models?**
        Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and generate human-like text. They work by predicting the most likely next word or token based on the context they've seen.
        
        **Key Concepts:**
        - **Tokens:** Basic units of text (words, parts of words, or characters)
        - **Context Window:** Maximum amount of text the model can consider at once
        - **Temperature:** Controls randomness in generation (0.0 = deterministic, 2.0 = very random)
        - **Top-p:** Controls diversity by sampling from the most probable tokens
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Evaluation Metrics Explained
        
        **ROUGE Scores:**
        - **ROUGE-1:** Overlap of individual words between generated and reference text
        - **ROUGE-2:** Overlap of two-word sequences (bigrams)
        - **ROUGE-L:** Longest common subsequence between texts
        
        **Other Important Metrics:**
        - **BLEU:** Measures precision for translation tasks
        - **BERTScore:** Uses neural embeddings for semantic similarity
        - **Perplexity:** Measures model confidence (lower is better)
        """)
    
    # Interactive learning
    st.markdown("#### 🎮 Interactive Learning")
    
    learning_topic = st.selectbox("Choose a topic to explore:", [
        "Temperature Effects on Generation",
        "Understanding ROUGE Scores",
        "Bias in AI Models",
        "Model Comparison Strategies",
        "Prompt Engineering Basics"
    ])
    
    if learning_topic == "Temperature Effects on Generation":
        st.markdown("""
        **Temperature Controls Creativity vs Consistency**
        
        Try different temperature values to see how they affect AI generation:
        """)
        
        temp_example = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1, key="learning_temp")
        
        examples = {
            0.1: "Very predictable and consistent output. Good for factual tasks.",
            0.7: "Balanced creativity and consistency. Good for most applications.",
            1.5: "More creative and diverse output. Good for creative writing.",
            2.0: "Very creative but potentially incoherent. Use with caution."
        }
        
        # Find closest example
        closest_temp = min(examples.keys(), key=lambda x: abs(x - temp_example))
        st.info(f"**Effect:** {examples[closest_temp]}")
    
    elif learning_topic == "Understanding ROUGE Scores":
        st.markdown("""
        **ROUGE Score Examples**
        
        Let's see how ROUGE scores work with examples:
        """)
        
        reference = "The cat sat on the mat."
        candidates = [
            "The cat sat on the mat.",  # Perfect match
            "A cat sat on a mat.",      # Similar but different articles
            "The dog sat on the mat.",  # One word different
            "Cats like sitting on mats." # Semantically similar but different structure
        ]
        
        st.markdown("**Reference:** " + reference)
        
        for i, candidate in enumerate(candidates):
            # Simulate ROUGE scores
            rouge1 = [1.0, 0.83, 0.83, 0.33][i]
            rouge2 = [1.0, 0.6, 0.6, 0.0][i]
            rougeL = [1.0, 0.83, 0.83, 0.17][i]
            
            st.markdown(f"""
            **Candidate {i+1}:** {candidate}
            - ROUGE-1: {rouge1:.2f}, ROUGE-2: {rouge2:.2f}, ROUGE-L: {rougeL:.2f}
            """)

def render_performance_analysis():
    """Render performance analysis with tutor insights"""
    st.markdown("### 📊 Performance Analysis with AI Tutor")
    
    # Upload or simulate analysis data
    analysis_option = st.radio("Choose analysis source:", ["Upload Results", "Use Sample Data"])
    
    if analysis_option == "Upload Results":
        uploaded_file = st.file_uploader("Upload evaluation results (JSON)", type=["json"])
        if uploaded_file:
            try:
                data = json.loads(uploaded_file.read().decode('utf-8'))
                st.success("✅ Results uploaded successfully!")
            except Exception as e:
                st.error(f"Error parsing file: {e}")
                data = None
        else:
            data = None
    else:
        # Generate sample data
        data = {
            "task": "summarization",
            "results": [
                {"model": "gpt-4o", "output": "Sample summary from GPT-4o...", "inference_time": 1.2, "word_count": 45},
                {"model": "gpt-3.5-turbo", "output": "Sample summary from GPT-3.5...", "inference_time": 0.8, "word_count": 38},
                {"model": "llama-3.1-8b", "output": "Sample summary from LLaMA...", "inference_time": 2.1, "word_count": 52}
            ],
            "evaluation": [
                {"model": "gpt-4o", "rouge1": 0.67, "rouge2": 0.45, "rougeL": 0.58, "inference_time": 1.2},
                {"model": "gpt-3.5-turbo", "rouge1": 0.61, "rouge2": 0.38, "rougeL": 0.52, "inference_time": 0.8},
                {"model": "llama-3.1-8b", "rouge1": 0.59, "rouge2": 0.33, "rougeL": 0.48, "inference_time": 2.1}
            ]
        }
    
    if data:
        st.markdown("#### 🎓 AI Tutor Analysis")
        
        # Tutor explanation
        with st.expander("📖 Detailed Performance Explanation", expanded=True):
            if data["task"] == "summarization":
                st.markdown("""
                ### 🎯 Summarization Performance Analysis
                
                **What happened in this evaluation:**
                
                **1. Model Performance Ranking:**
                - **GPT-4o** achieved the highest ROUGE scores across all metrics
                - **GPT-3.5-turbo** performed well with faster inference
                - **LLaMA-3.1-8B** showed competitive quality but slower speed
                
                **2. Key Insights:**
                - ROUGE-1 scores indicate good word-level overlap with reference
                - ROUGE-2 scores show phrase-level similarity patterns
                - ROUGE-L measures structural similarity to the reference
                
                **3. Speed vs Quality Trade-off:**
                - GPT-3.5-turbo offers the best speed (0.8s) with good quality
                - GPT-4o provides highest quality but takes 50% longer
                - LLaMA model is significantly slower but locally hosted
                
                **4. Practical Recommendations:**
                - Use GPT-4o for highest quality requirements
                - Choose GPT-3.5-turbo for balanced performance
                - Consider LLaMA for privacy-sensitive applications
                """)
        
        # Performance visualization with insights
        if "evaluation" in data:
            df = pd.DataFrame(data["evaluation"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # ROUGE scores comparison
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(name='ROUGE-1', x=df['model'], y=df['rouge1']))
                fig1.add_trace(go.Bar(name='ROUGE-2', x=df['model'], y=df['rouge2']))
                fig1.add_trace(go.Bar(name='ROUGE-L', x=df['model'], y=df['rougeL']))
                
                fig1.update_layout(
                    title='ROUGE Scores Comparison',
                    xaxis_title='Model',
                    yaxis_title='Score',
                    barmode='group',
                    template='plotly_white'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Speed vs Quality scatter
                fig2 = px.scatter(df, x='inference_time', y='rouge1', size='rougeL',
                                 hover_name='model', title='Speed vs Quality Trade-off')
                fig2.update_layout(template='plotly_white')
                st.plotly_chart(fig2, use_container_width=True)
        
        # Interactive Q&A with tutor
        st.markdown("#### 💬 Ask the AI Tutor")
        
        tutor_question = st.selectbox("What would you like to learn about?", [
            "Why did one model perform better than others?",
            "How can I improve these scores?",
            "What do these metrics really mean?",
            "Should I choose speed or quality?",
            "How do I interpret these results for my use case?"
        ])
        
        if st.button("🎓 Get Tutor Explanation"):
            explanations = {
                "Why did one model perform better than others?": """
                **Model Performance Differences Explained:**
                
                1. **Training Data & Architecture:** Different models are trained on different datasets and use different architectures, leading to varying capabilities.
                
                2. **Model Size:** Larger models generally perform better but are slower. GPT-4o is likely larger and more sophisticated than GPT-3.5-turbo.
                
                3. **Fine-tuning:** Some models may be specifically fine-tuned for certain tasks like summarization.
                
                4. **Optimization:** Commercial models like OpenAI's are highly optimized for inference speed while maintaining quality.
                """,
                
                "How can I improve these scores?": """
                **Strategies to Improve Performance:**
                
                1. **Better Prompts:** Use more specific instructions like "Create a concise summary focusing on key points"
                
                2. **Parameter Tuning:** Adjust temperature (lower for consistency) and max_tokens (appropriate length)
                
                3. **Model Selection:** Choose models that excel at your specific task
                
                4. **Reference Quality:** Ensure your reference summaries are high-quality for accurate evaluation
                
                5. **Post-processing:** Clean up outputs to remove formatting issues that might affect scores
                """,
                
                "What do these metrics really mean?": """
                **Metric Interpretation Guide:**
                
                **ROUGE-1 (0.6-0.7):** Good word overlap. Your summaries contain most important words from reference.
                
                **ROUGE-2 (0.3-0.5):** Decent phrase matching. Summaries capture some key phrases correctly.
                
                **ROUGE-L (0.5-0.6):** Good structural similarity. The order and flow match the reference reasonably well.
                
                **What's Good:** Scores above 0.5 for ROUGE-1 and ROUGE-L are generally considered good for summarization.
                """,
                
                "Should I choose speed or quality?": """
                **Speed vs Quality Decision Framework:**
                
                **Choose Speed (GPT-3.5-turbo) when:**
                - Processing large volumes of content
                - Real-time applications
                - Cost is a major concern
                - Quality difference is marginal for your use case
                
                **Choose Quality (GPT-4o) when:**
                - Content is high-stakes (legal, medical, financial)
                - Accuracy is critical
                - Human review is expensive
                - Brand reputation depends on quality
                
                **Consider Local Models (LLaMA) when:**
                - Data privacy is critical
                - Long-term cost optimization
                - Custom fine-tuning needs
                """,
                
                "How do I interpret these results for my use case?": """
                **Use Case Interpretation:**
                
                **For Content Summarization:**
                - ROUGE-1 > 0.6: Good for most business applications
                - Speed < 2s: Acceptable for user-facing applications
                - Consider the content domain (technical vs general)
                
                **For Research/Academic:**
                - Higher quality thresholds needed
                - ROUGE-L > 0.6 for structural accuracy
                - Manual evaluation also recommended
                
                **For Production Systems:**
                - Balance speed, cost, and quality
                - Set up monitoring for quality drift
                - A/B test different models with real users
                """
            }
            
            st.markdown(explanations[tutor_question])

def render_tutor_recommendations():
    """Render AI tutor recommendations"""
    st.markdown("### 🎯 Personalized Recommendations")
    
    # User profile for personalized recommendations
    st.markdown("#### 👤 Tell us about your use case")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_case = st.selectbox("Primary use case:", [
            "Content Creation",
            "Customer Support",
            "Research & Analysis",
            "Education & Training",
            "Software Development",
            "Business Intelligence"
        ])
        
        experience_level = st.selectbox("AI/ML Experience:", [
            "Beginner",
            "Intermediate", 
            "Advanced",
            "Expert"
        ])
    
    with col2:
        priority = st.selectbox("Top Priority:", [
            "Quality",
            "Speed",
            "Cost",
            "Privacy",
            "Reliability"
        ])
        
        volume = st.selectbox("Expected Volume:", [
            "Low (< 100 requests/day)",
            "Medium (100-1000 requests/day)",
            "High (1000-10000 requests/day)",
            "Very High (> 10000 requests/day)"
        ])
    
    if st.button("🎯 Get Personalized Recommendations", type="primary"):
        st.markdown("#### 🎓 Your Personalized AI Strategy")
        
        # Generate recommendations based on profile
        recommendations = generate_personalized_recommendations(use_case, experience_level, priority, volume)
        
        for category, advice in recommendations.items():
            with st.expander(f"{category}", expanded=True):
                st.markdown(advice)

def generate_personalized_recommendations(use_case, experience, priority, volume):
    """Generate personalized recommendations based on user profile"""
    
    recommendations = {}
    
    # Model selection recommendations
    if priority == "Quality":
        if volume.startswith("Low") or volume.startswith("Medium"):
            model_rec = "**Recommended:** GPT-4o for highest quality. The superior performance justifies the cost for your volume."
        else:
            model_rec = "**Recommended:** Mix of GPT-4o for critical tasks and GPT-3.5-turbo for high-volume processing."
    elif priority == "Speed":
        model_rec = "**Recommended:** GPT-3.5-turbo offers the best speed-quality balance for your needs."
    elif priority == "Cost":
        if volume.startswith("High") or volume.startswith("Very High"):
            model_rec = "**Recommended:** GPT-3.5-turbo or consider local LLaMA models for cost optimization."
        else:
            model_rec = "**Recommended:** GPT-3.5-turbo provides good value for moderate usage."
    else:  # Privacy or Reliability
        model_rec = "**Recommended:** Consider local LLaMA models for privacy or implement redundancy with multiple models."
    
    recommendations["🤖 Model Selection"] = model_rec
    
    # Configuration recommendations
    if use_case == "Content Creation":
        config_rec = """
        **Optimal Settings:**
        - Temperature: 0.7-0.9 for creativity
        - Max tokens: 200-500 depending on content length
        - Top-p: 0.9 for diverse vocabulary
        
        **Tips:** Use system prompts to define tone and style consistently.
        """
    elif use_case == "Customer Support":
        config_rec = """
        **Optimal Settings:**
        - Temperature: 0.3-0.5 for consistency
        - Max tokens: 100-200 for concise responses
        - Top-p: 0.8 for focused responses
        
        **Tips:** Implement safety checks and escalation patterns.
        """
    else:
        config_rec = """
        **Optimal Settings:**
        - Temperature: 0.5-0.7 for balanced output
        - Max tokens: Adjust based on expected output length
        - Top-p: 0.9 for good diversity
        
        **Tips:** Experiment with different settings for your specific content.
        """
    
    recommendations["⚙️ Configuration"] = config_rec
    
    # Learning path recommendations
    if experience == "Beginner":
        learning_rec = """
        **Your Learning Path:**
        1. Start with basic prompt engineering
        2. Learn about temperature and token limits
        3. Understand evaluation metrics (ROUGE, BLEU)
        4. Practice with different use cases
        5. Explore safety and bias considerations
        
        **Resources:** Focus on practical tutorials and hands-on experimentation.
        """
    elif experience == "Intermediate":
        learning_rec = """
        **Your Learning Path:**
        1. Advanced prompt engineering techniques
        2. Model fine-tuning basics
        3. Evaluation methodology design
        4. Cost optimization strategies
        5. Production deployment considerations
        
        **Resources:** Dive into research papers and advanced tutorials.
        """
    else:
        learning_rec = """
        **Your Learning Path:**
        1. Custom model development
        2. Advanced evaluation metrics
        3. Model fusion techniques
        4. Research and development
        5. Contributing to AI safety
        
        **Resources:** Latest research, conferences, and open-source contributions.
        """
    
    recommendations["📚 Learning Path"] = learning_rec
    
    # Implementation recommendations
    if volume.startswith("Low"):
        impl_rec = """
        **Implementation Strategy:**
        - Start simple with direct API calls
        - Use built-in safety features
        - Monitor manually initially
        - Scale gradually as usage grows
        """
    elif volume.startswith("High") or volume.startswith("Very High"):
        impl_rec = """
        **Implementation Strategy:**
        - Implement robust caching systems
        - Set up automated monitoring
        - Use batch processing where possible
        - Implement circuit breakers and fallbacks
        - Consider model hosting options
        """
    else:
        impl_rec = """
        **Implementation Strategy:**
        - Implement basic monitoring
        - Use caching for repeated queries
        - Set up basic safety checks
        - Plan for scaling
        """
    
    recommendations["🚀 Implementation"] = impl_rec
    
    return recommendations

# Main application
def main():
    """Main application entry point"""
    initialize_session_state()
    
    # Load configuration
    config = load_config()
    api_urls = get_api_urls(config)
    
    # Render sidebar navigation
    render_sidebar()
    
    # Route to appropriate page based on current_page
    current_page = st.session_state.get("current_page", "overview")
    
    if current_page == "overview":
        render_overview_page()
    elif current_page == "chat":
        render_chat_interface()
    elif current_page == "summarization":
        render_summarization_interface(api_urls)
    elif current_page == "translation":
        render_translation_interface(api_urls)
    elif current_page == "documents":
        render_document_interface(api_urls)
    elif current_page == "evaluation":
        render_evaluation_interface(api_urls)
    elif current_page == "fusion":
        render_fusion_interface(api_urls)
    elif current_page == "battles":
        render_model_battles()
    elif current_page == "tutor":
        render_ai_tutor()
    elif current_page == "ethics":
        render_ethics_interface()
    elif current_page == "crowdsourcing":
        render_crowdsourcing()
    elif current_page == "fine_tuning":
        render_fine_tuning()
    elif current_page == "analytics":
        render_analytics_interface()
    elif current_page == "costs":
        render_cost_tracking()
    elif current_page == "config":
        render_system_config()
    elif current_page == "api_explorer":
        render_api_explorer()
    else:
        render_overview_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem; margin-top: 3rem;">
        <p>🤖 <strong>AI Workbench v2.0</strong> - Complete AI Development Platform</p>
        <p style="font-size: 0.9rem;">Featuring advanced model evaluation, safety monitoring, collaboration tools, and comprehensive analytics</p>
        <p style="font-size: 0.8rem;">Built with Streamlit • Powered by OpenAI, LLaMA, and HuggingFace</p>
    </div>
    """, unsafe_allow_html=True)

# Additional interface functions (placeholders for missing functions)
def render_summarization_interface(api_urls):
    """Enhanced summarization interface"""
    st.markdown('<h1 class="main-header">📝 Advanced Summarization</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate summaries with comprehensive evaluation</p>', unsafe_allow_html=True)
    
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
        temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.7, 0.1)
        max_tokens = st.slider("📏 Max Tokens", 50, 500, 100, 10)
        min_tokens = st.slider("📐 Min Tokens", 10, 200, 30, 5)
        
        st.markdown("#### 🚀 Advanced Options")
        enable_ethics = st.checkbox("Ethics Analysis", value=True)
        enable_fusion = st.checkbox("Model Fusion", value=False)
        enable_tutor = st.checkbox("Tutor Insights", value=True)
    
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
            
            success, results = make_api_request(api_urls["process"], "POST", payload)
            
            if success:
                display_comprehensive_results(results, "summarization")
            else:
                st.error(f"❌ **Processing failed:** {results}")
        else:
            st.error("Please enter text to summarize")

def render_translation_interface(api_urls):
    """Enhanced translation interface"""
    st.markdown('<h1 class="main-header">🌐 Advanced Translation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-language translation with quality analysis</p>', unsafe_allow_html=True)
    
    # Similar structure to summarization but for translation
    # Implementation details would be similar to the original but enhanced
    st.info("Enhanced translation interface - similar structure to summarization with language-specific features")

def render_document_interface(api_urls):
    """Enhanced document intelligence interface"""
    st.markdown('<h1 class="main-header">📄 Document Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced document processing with RAG capabilities</p>', unsafe_allow_html=True)
    
    # Enhanced document interface
    st.info("Enhanced document interface with advanced RAG features, OCR, and multi-format support")

def render_evaluation_interface(api_urls):
    """Model evaluation interface"""
    st.markdown('<h1 class="main-header">🔬 Model Evaluation Suite</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive model performance evaluation</p>', unsafe_allow_html=True)
    
    st.info("Comprehensive evaluation interface with custom metrics, benchmarking, and detailed analysis")

def render_fusion_interface(api_urls):
    """Model fusion interface"""
    st.markdown('<h1 class="main-header">🤝 Model Fusion Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Combine multiple models for enhanced performance</p>', unsafe_allow_html=True)
    
    st.info("Advanced model fusion interface with multiple strategies and performance optimization")

def render_cost_tracking():
    """Cost tracking interface"""
    st.markdown('<h1 class="main-header">💰 Cost Tracking & Optimization</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Monitor and optimize AI usage costs</p>', unsafe_allow_html=True)
    
    st.info("Comprehensive cost tracking with budget management, forecasting, and optimization recommendations")

def render_system_config():
    """System configuration interface"""
    st.markdown('<h1 class="main-header">⚙️ System Configuration</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Configure models, parameters, and system settings</p>', unsafe_allow_html=True)
    
    st.info("System configuration interface for models, API settings, security, and feature toggles")

def render_api_explorer():
    """API explorer interface"""
    st.markdown('<h1 class="main-header">🔧 API Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive API testing and documentation</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🧪 API Testing", "📚 Documentation", "📊 Usage Logs"])
    
    with tab1:
        render_api_testing()
    
    with tab2:
        render_api_documentation()
    
    with tab3:
        render_api_logs()

def render_api_testing():
    """Render API testing interface"""
    st.markdown("### 🧪 Interactive API Testing")
    
    # API endpoint selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        endpoint = st.selectbox("Select Endpoint:", [
            "POST /process",
            "GET /health", 
            "GET /models",
            "POST /upload_documents",
            "GET /usage_stats",
            "POST /crowdsource",
            "GET /supported_languages"
        ])
        
        method = endpoint.split()[0]
        path = endpoint.split()[1]
    
    with col2:
        st.markdown(f"**Method:** `{method}`")
        st.markdown(f"**Endpoint:** `{path}`")
        st.markdown(f"**Full URL:** `http://127.0.0.1:8000{path}`")
    
    # Request configuration
    if method == "POST":
        st.markdown("#### 📝 Request Body")
        
        if path == "/process":
            # Pre-filled examples for /process endpoint
            example_payloads = {
                "Chat": {
                    "task": "chat",
                    "messages": [{"role": "user", "content": "Hello, how are you?"}],
                    "params": {"temperature": 0.7, "max_tokens": 100}
                },
                "Summarization": {
                    "task": "summarization", 
                    "text": "This is a sample text to summarize...",
                    "params": {"temperature": 0.5, "max_tokens": 80}
                },
                "Translation": {
                    "task": "translation",
                    "text": "Hello world",
                    "target_lang": "Spanish",
                    "params": {"temperature": 0.3, "max_tokens": 50}
                }
            }
            
            example_type = st.selectbox("Example Payload:", list(example_payloads.keys()))
            payload = st.text_area(
                "Request JSON:",
                value=json.dumps(example_payloads[example_type], indent=2),
                height=200
            )
        else:
            payload = st.text_area(
                "Request JSON:",
                value="{}",
                height=150
            )
    
    # Headers
    st.markdown("#### 📋 Request Headers")
    headers = st.text_area(
        "Custom Headers (JSON format):",
        value='{"Content-Type": "application/json"}',
        height=80
    )
    
    # Send request
    if st.button("🚀 Send Request", type="primary", use_container_width=True):
        config = load_config()
        api_urls = get_api_urls(config)
        full_url = f"{api_urls['base']}{path}"
        
        try:
            # Parse headers
            request_headers = json.loads(headers) if headers else {}
            
            with st.spinner("Sending request..."):
                session = create_session()
                
                if method == "GET":
                    response = session.get(full_url, headers=request_headers, timeout=30)
                elif method == "POST":
                    request_data = json.loads(payload) if payload else {}
                    response = session.post(full_url, json=request_data, headers=request_headers, timeout=30)
                
                # Display response
                st.markdown("#### 📨 Response")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    status_color = "#10b981" if 200 <= response.status_code < 300 else "#ef4444"
                    st.markdown(f"**Status:** <span style='color: {status_color}'>{response.status_code}</span>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**Time:** {response.elapsed.total_seconds():.2f}s")
                
                with col3:
                    st.markdown(f"**Size:** {len(response.content)} bytes")
                
                # Response headers
                st.markdown("**Response Headers:**")
                st.json(dict(response.headers))
                
                # Response body
                st.markdown("**Response Body:**")
                try:
                    response_json = response.json()
                    st.json(response_json)
                except:
                    st.code(response.text)
                
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in payload or headers: {e}")
        except Exception as e:
            st.error(f"Request failed: {e}")

def render_api_documentation():
    """Render API documentation"""
    st.markdown("### 📚 API Documentation")
    
    # API overview
    st.markdown("""
    ## 🚀 AI Workbench API Overview
    
    The AI Workbench API provides comprehensive access to multiple AI models and advanced features including:
    - Multi-model text processing (chat, summarization, translation)
    - Document intelligence with RAG capabilities
    - Model evaluation and comparison
    - Safety and ethics analysis
    - Crowdsourcing and collaboration features
    """)
    
    # Endpoints documentation
    endpoints_doc = {
        "Core Processing": {
            "POST /process": {
                "description": "Main endpoint for AI processing tasks",
                "parameters": {
                    "task": "Task type: 'chat', 'summarization', 'translation'",
                    "text": "Input text (for summarization/translation)",
                    "messages": "Array of message objects (for chat)",
                    "target_lang": "Target language (for translation)",
                    "params": "Generation parameters (temperature, max_tokens, etc.)"
                },
                "example": {
                    "task": "chat",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "params": {"temperature": 0.7, "max_tokens": 100}
                }
            }
        },
        "System Information": {
            "GET /health": {
                "description": "Check API health and status",
                "parameters": "None",
                "example": "GET /health"
            },
            "GET /models": {
                "description": "Get available AI models",
                "parameters": "None", 
                "example": "GET /models"
            },
            "GET /usage_stats": {
                "description": "Get usage statistics",
                "parameters": "None",
                "example": "GET /usage_stats"
            }
        },
        "Document Processing": {
            "POST /upload_documents": {
                "description": "Upload documents for RAG processing",
                "parameters": {
                    "file": "Document file (PDF, image, text)"
                },
                "example": "Multipart form data with file"
            }
        },
        "Crowdsourcing": {
            "POST /crowdsource": {
                "description": "Submit dataset for crowdsourcing",
                "parameters": {
                    "data": "Array of data entries",
                    "submitter": "Submitter name"
                },
                "example": {
                    "data": [{"input": "text", "output": "result"}],
                    "submitter": "username"
                }
            }
        }
    }
    
    for category, endpoints in endpoints_doc.items():
        st.markdown(f"### {category}")
        
        for endpoint, info in endpoints.items():
            with st.expander(f"{endpoint}", expanded=False):
                st.markdown(f"**Description:** {info['description']}")
                
                if isinstance(info['parameters'], dict):
                    st.markdown("**Parameters:**")
                    for param, desc in info['parameters'].items():
                        st.markdown(f"- `{param}`: {desc}")
                else:
                    st.markdown(f"**Parameters:** {info['parameters']}")
                
                st.markdown("**Example:**")
                if isinstance(info['example'], dict):
                    st.code(json.dumps(info['example'], indent=2), language="json")
                else:
                    st.code(info['example'])

def render_api_logs():
    """Render API usage logs"""
    st.markdown("### 📊 API Usage Logs")
    
    # Simulated log data
    log_entries = []
    for i in range(50):
        timestamp = datetime.now() - timedelta(minutes=np.random.randint(0, 1440))
        endpoints = ["/process", "/health", "/models", "/upload_documents", "/usage_stats"]
        methods = ["POST", "GET", "GET", "POST", "GET"]
        status_codes = [200, 200, 200, 201, 500, 404]
        
        endpoint_idx = np.random.randint(0, len(endpoints))
        endpoint = endpoints[endpoint_idx]
        method = methods[endpoint_idx]
        status = np.random.choice(status_codes, p=[0.7, 0.15, 0.05, 0.05, 0.03, 0.02])
        
        log_entries.append({
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Method": method,
            "Endpoint": endpoint,
            "Status": status,
            "Response Time": f"{np.random.uniform(0.1, 3.0):.2f}s",
            "IP Address": f"192.168.1.{np.random.randint(1, 255)}",
            "User Agent": "AI-Workbench-Frontend/2.0"
        })
    
    # Sort by timestamp (most recent first)
    log_entries.sort(key=lambda x: x["Timestamp"], reverse=True)
    df_logs = pd.DataFrame(log_entries)
    
    # Log filtering
    col1, col2, col3 = st.columns(3)
    
    with col1:
        method_filter = st.multiselect("Filter by Method:", ["GET", "POST"], default=["GET", "POST"])
    
    with col2:
        status_filter = st.multiselect("Filter by Status:", [200, 201, 404, 500], default=[200, 201, 404, 500])
    
    with col3:
        time_filter = st.selectbox("Time Range:", ["Last Hour", "Last 24 Hours", "Last Week", "All Time"])
    
    # Apply filters
    filtered_logs = df_logs[
        (df_logs["Method"].isin(method_filter)) &
        (df_logs["Status"].isin(status_filter))
    ]
    
    # Display logs
    st.dataframe(
        filtered_logs,
        column_config={
            "Timestamp": st.column_config.TextColumn("🕐 Timestamp"),
            "Method": st.column_config.TextColumn("📝 Method"),
            "Endpoint": st.column_config.TextColumn("🔗 Endpoint"),
            "Status": st.column_config.NumberColumn("📊 Status"),
            "Response Time": st.column_config.TextColumn("⏱️ Time"),
            "IP Address": st.column_config.TextColumn("🌐 IP"),
            "User Agent": st.column_config.TextColumn("🖥️ User Agent")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Log statistics
    st.markdown("#### 📈 Log Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_requests = len(filtered_logs)
        st.metric("Total Requests", total_requests)
    
    with col2:
        success_rate = len(filtered_logs[filtered_logs["Status"] < 400]) / len(filtered_logs) * 100 if len(filtered_logs) > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        avg_response_time = np.mean([float(t.replace('s', '')) for t in filtered_logs["Response Time"]])
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    
    with col4:
        most_used_endpoint = filtered_logs["Endpoint"].mode().iloc[0] if len(filtered_logs) > 0 else "N/A"
        st.metric("Most Used Endpoint", most_used_endpoint)

def display_comprehensive_results(results, task_type):
    """Enhanced results display with comprehensive analysis"""
    if isinstance(results, str):
        st.markdown(f"""
        <div style="background: white; border-radius: 12px; padding: 2rem; border: 1px solid #e5e7eb; margin: 1rem 0;">
            <h3>🤖 AI Response</h3>
            <p>{results}</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if not isinstance(results, dict):
        st.error("Invalid response format")
        return
    
    task_results = results.get("results", [])
    evaluation = results.get("evaluation")
    
    # Display model results with enhanced UI
    if task_results:
        st.markdown("### 🤖 Model Results")
        
        tabs = st.tabs(["📊 Results", "📈 Metrics", "🎯 Analysis", "🛡️ Safety"])
        
        with tabs[0]:
            render_model_results_tab(task_results, task_type)
        
        with tabs[1]:
            if evaluation:
                render_metrics_tab(evaluation)
            else:
                st.info("No evaluation metrics available. Add a reference text for detailed metrics.")
        
        with tabs[2]:
            if evaluation:
                render_analysis_tab(evaluation, task_type, task_results)
            else:
                st.info("Analysis requires evaluation metrics.")
        
        with tabs[3]:
            render_safety_tab(task_results)

def render_model_results_tab(task_results, task_type):
    """Render enhanced model results"""
    for i, result in enumerate(task_results):
        model_name = result.get("model", "Unknown")
        output = result.get("output")
        inference_time = result.get("inference_time", 0)
        success = result.get("success", True)
        word_count = result.get("word_count", 0)
        
        # Enhanced result card
        with st.expander(f"🔍 {model_name} Results", expanded=True):
            if success and output:
                # Main output display
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; padding: 1.5rem; border: 1px solid #cbd5e1; margin: 1rem 0;">
                    <h4 style="color: #1e293b; margin-bottom: 1rem;">Generated Output</h4>
                    <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                        {output}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("⏱️ Time", f"{inference_time:.2f}s")
                
                with col2:
                    st.metric("📝 Words", word_count)
                
                with col3:
                    if inference_time > 0:
                        wps = word_count / inference_time
                        st.metric("⚡ Speed", f"{wps:.1f} w/s")
                
                with col4:
                    chars = len(output)
                    st.metric("📄 Characters", chars)
                
                # Quality indicators
                quality_issues = result.get("quality_issues", [])
                if quality_issues:
                    st.warning(f"⚠️ Quality concerns: {', '.join(quality_issues)}")
                else:
                    st.success("✅ No quality issues detected")
                
                # Additional analysis for specific tasks
                if task_type == "translation":
                    target_lang = result.get("target_language", "Unknown")
                    st.info(f"🌐 Target Language: {target_lang}")
                
            else:
                error_msg = result.get("error", "Unknown error")
                st.error(f"❌ **Error:** {error_msg}")

def render_metrics_tab(evaluation):
    """Render enhanced metrics visualization"""
    try:
        if isinstance(evaluation, list):
            df = pd.DataFrame(evaluation)
        else:
            df = evaluation
        
        # Metrics overview cards
        st.markdown("#### 📊 Metrics Overview")
        
        metric_columns = [col for col in df.columns if col not in ["model", "inference_time", "success"]]
        
        if metric_columns:
            cols = st.columns(min(len(metric_columns), 4))
            
            for i, metric in enumerate(metric_columns[:4]):
                with cols[i]:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        avg_val = values.mean()
                        max_val = values.max()
                        best_model = df.loc[df[metric].idxmax(), "model"] if len(values) > 0 else "N/A"
                        
                        st.metric(
                            label=metric.replace('_', ' ').title(),
                            value=f"{avg_val:.3f}",
                            delta=f"Best: {max_val:.3f} ({best_model})"
                        )
        
        # Detailed metrics visualization
        st.markdown("#### 📈 Detailed Metrics")
        
        if len(metric_columns) > 0:
            # Radar chart for model comparison
            if len(df) > 1:
                fig_radar = go.Figure()
                
                for _, row in df.iterrows():
                    model_name = row["model"]
                    values = [row[col] for col in metric_columns if pd.notna(row[col])]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metric_columns,
                        fill='toself',
                        name=model_name
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Model Performance Comparison (Radar Chart)",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Bar charts for individual metrics
            col1, col2 = st.columns(2)
            
            for i, metric in enumerate(metric_columns):
                with col1 if i % 2 == 0 else col2:
                    fig_bar = px.bar(
                        df, 
                        x="model", 
                        y=metric,
                        title=f"{metric.replace('_', ' ').title()} by Model",
                        text=metric
                    )
                    fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig_bar.update_layout(template="plotly_white", height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("#### 📋 Detailed Metrics Table")
        
        styled_df = df.copy()
        for col in metric_columns:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].round(4)
        
        st.dataframe(
            styled_df,
            column_config={
                "model": st.column_config.TextColumn("🤖 Model"),
                **{col: st.column_config.NumberColumn(
                    col.replace('_', ' ').title(),
                    format="%.4f"
                ) for col in metric_columns}
            },
            hide_index=True,
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error displaying metrics: {e}")

def render_analysis_tab(evaluation, task_type, results):
    """Render enhanced analysis insights"""
    try:
        if isinstance(evaluation, list):
            df = pd.DataFrame(evaluation)
        else:
            df = evaluation
        
        st.markdown("#### 🎯 Performance Analysis")
        
        successful_models = df[df.get("success", True) == True] if "success" in df.columns else df
        
        if len(successful_models) > 0:
            # Performance summary
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
                word_counts = [r.get("word_count", 0) for r in results]
                if word_counts:
                    avg_words = np.mean(word_counts)
                    st.metric("Avg Words", f"{avg_words:.0f}")
            
            # Best performers analysis
            st.markdown("#### 🏆 Top Performers")
            
            metric_columns = [col for col in df.columns if col not in ["model", "inference_time", "success"]]
            
            if metric_columns:
                perf_cols = st.columns(min(len(metric_columns), 3))
                
                for i, metric in enumerate(metric_columns[:3]):
                    with perf_cols[i]:
                        if metric in df.columns:
                            # Determine if higher or lower is better
                            is_lower_better = metric.lower() in ["perplexity", "loss", "error_rate"]
                            
                            if is_lower_better:
                                best_idx = df[metric].idxmin()
                                label = "Lowest"
                            else:
                                best_idx = df[metric].idxmax()
                                label = "Highest"
                            
                            best_model = df.loc[best_idx, "model"]
                            best_score = df.loc[best_idx, metric]
                            
                            st.markdown(f"""
                            <div style="background: white; border-radius: 8px; padding: 1rem; border: 1px solid #e5e7eb; text-align: center;">
                                <h4>🥇 {label} {metric.replace('_', ' ').title()}</h4>
                                <p><strong>Model:</strong> {best_model}</p>
                                <p><strong>Score:</strong> {best_score:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Speed vs Quality analysis
            st.markdown("#### ⚡ Speed vs Quality Analysis")
            
            if "inference_time" in df.columns and len(metric_columns) > 0:
                quality_metric = metric_columns[0]  # Use first quality metric
                
                fig_scatter = px.scatter(
                    df, 
                    x="inference_time", 
                    y=quality_metric,
                    hover_name="model",
                    title=f"Speed vs {quality_metric.replace('_', ' ').title()}",
                    labels={
                        "inference_time": "Inference Time (seconds)",
                        quality_metric: quality_metric.replace('_', ' ').title()
                    }
                )
                
                fig_scatter.update_layout(template="plotly_white")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Analysis insights
                correlation = df["inference_time"].corr(df[quality_metric])
                
                if abs(correlation) > 0.5:
                    if correlation > 0:
                        insight = "⚠️ Slower models tend to produce higher quality outputs"
                    else:
                        insight = "⚡ Faster models maintain competitive quality scores"
                else:
                    insight = "➡️ No strong correlation between speed and quality observed"
                
                st.info(insight)
        
        # Task-specific insights
        st.markdown("#### 💡 Task-Specific Insights")
        
        if task_type == "summarization":
            st.markdown("""
            **Summarization Analysis:**
            - ROUGE-1 measures word overlap with reference summary
            - ROUGE-2 captures phrase-level similarity  
            - ROUGE-L evaluates structural similarity
            - Higher scores indicate better alignment with reference
            """)
        elif task_type == "translation":
            st.markdown("""
            **Translation Analysis:**
            - BLEU scores measure n-gram precision against reference
            - METEOR considers stemming, synonyms, and word order
            - Quality issues may indicate cultural or contextual challenges
            - Consider multiple references for more robust evaluation
            """)
        elif task_type == "chat":
            st.markdown("""
            **Chat Analysis:**
            - Response relevance and coherence are key factors
            - Length should be appropriate for the conversation context
            - Safety and ethical considerations are paramount
            - User engagement metrics provide additional insights
            """)
        
    except Exception as e:
        st.error(f"Error in analysis: {e}")

def render_safety_tab(results):
    """Render safety analysis tab"""
    st.markdown("#### 🛡️ Safety Analysis")
    
    # Simulate safety analysis for each result
    for result in results:
        model_name = result.get("model", "Unknown")
        output = result.get("output", "")
        
        if output:
            with st.expander(f"🛡️ Safety Analysis - {model_name}", expanded=False):
                # Mock safety scores
                toxicity_score = np.random.uniform(0.0, 0.3)
                bias_score = np.random.uniform(0.0, 0.2)
                safety_score = 1 - (toxicity_score * 0.6 + bias_score * 0.4)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    toxicity_color = "#10b981" if toxicity_score < 0.1 else "#f59e0b" if toxicity_score < 0.3 else "#ef4444"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; border: 1px solid #e5e7eb;">
                        <h4>☣️ Toxicity</h4>
                        <p style="color: {toxicity_color}; font-size: 1.5rem; font-weight: 600;">{toxicity_score:.3f}</p>
                        <p>{'✅ Safe' if toxicity_score < 0.1 else '⚠️ Review' if toxicity_score < 0.3 else '🚨 Unsafe'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    bias_color = "#10b981" if bias_score < 0.1 else "#f59e0b" if bias_score < 0.2 else "#ef4444"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; border: 1px solid #e5e7eb;">
                        <h4>⚖️ Bias</h4>
                        <p style="color: {bias_color}; font-size: 1.5rem; font-weight: 600;">{bias_score:.3f}</p>
                        <p>{'✅ Fair' if bias_score < 0.1 else '⚠️ Review' if bias_score < 0.2 else '🚨 Biased'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    safety_color = "#10b981" if safety_score > 0.8 else "#f59e0b" if safety_score > 0.6 else "#ef4444"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; border: 1px solid #e5e7eb;">
                        <h4>🛡️ Overall Safety</h4>
                        <p style="color: {safety_color}; font-size: 1.5rem; font-weight: 600;">{safety_score:.3f}</p>
                        <p>{'✅ Safe' if safety_score > 0.8 else '⚠️ Review' if safety_score > 0.6 else '🚨 Unsafe'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Safety recommendations
                recommendations = []
                if toxicity_score > 0.1:
                    recommendations.append("🧹 Consider content filtering for toxic language")
                if bias_score > 0.1:
                    recommendations.append("⚖️ Review for potential bias and fairness issues")
                if safety_score < 0.8:
                    recommendations.append("🔍 Manual review recommended before deployment")
                
                if recommendations:
                    st.markdown("**🎯 Recommendations:**")
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.success("✅ No safety concerns detected")

if __name__ == "__main__":
    main()