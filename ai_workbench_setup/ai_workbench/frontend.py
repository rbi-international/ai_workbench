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
    page_title="AI Workbench - LLaMA vs OpenAI Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for voice interface and charts
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
    border-left: 4px solid #007bff;
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

/* Chart Styling */
.plotly-graph-div {
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin: 10px 0;
}

.metric-summary {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}

.ranking-item {
    background: #f8f9fa;
    padding: 8px 12px;
    border-left: 4px solid #007bff;
    margin: 5px 0;
    border-radius: 0 5px 5px 0;
}

.comparison-winner {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    padding: 10px;
    border-radius: 8px;
    border-left: 5px solid #28a745;
    margin: 10px 0;
}

.comparison-loser {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    padding: 10px;
    border-radius: 8px;
    border-left: 5px solid #dc3545;
    margin: 10px 0;
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

.model-comparison-header {
    background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 8px 25px rgba(108, 92, 231, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Comprehensive metrics definitions for LLaMA vs OpenAI comparison
METRICS_DEFINITIONS = {
    # Text Quality Metrics
    "rouge1": {
        "name": "ROUGE-1",
        "description": "Unigram overlap between generated and reference text",
        "range": "0-1 (higher is better)",
        "good_threshold": 0.4,
        "category": "Text Quality"
    },
    "rouge2": {
        "name": "ROUGE-2", 
        "description": "Bigram overlap between generated and reference text",
        "range": "0-1 (higher is better)",
        "good_threshold": 0.25,
        "category": "Text Quality"
    },
    "rougeL": {
        "name": "ROUGE-L",
        "description": "Longest common subsequence between texts",
        "range": "0-1 (higher is better)", 
        "good_threshold": 0.35,
        "category": "Text Quality"
    },
    "bertscore_f1": {
        "name": "BERTScore F1",
        "description": "Semantic similarity using BERT embeddings",
        "range": "0-1 (higher is better)",
        "good_threshold": 0.7,
        "category": "Text Quality"
    },
    "bleu": {
        "name": "BLEU",
        "description": "N-gram precision for translation quality",
        "range": "0-1 (higher is better)",
        "good_threshold": 0.3,
        "category": "Translation Quality"
    },
    "meteor": {
        "name": "METEOR",
        "description": "Translation evaluation with word order and synonyms",
        "range": "0-1 (higher is better)",
        "good_threshold": 0.4,
        "category": "Translation Quality"
    },
    
    # Performance Metrics
    "inference_time": {
        "name": "Inference Time",
        "description": "Time taken to generate response",
        "range": "0+ seconds (lower is better)",
        "good_threshold": 3.0,
        "category": "Performance"
    },
    "tokens_per_second": {
        "name": "Tokens/Second",
        "description": "Generation speed in tokens per second",
        "range": "0+ (higher is better)",
        "good_threshold": 50,
        "category": "Performance"
    },
    "memory_usage": {
        "name": "Memory Usage",
        "description": "GPU/RAM memory consumption in GB",
        "range": "0+ GB (lower is better)",
        "good_threshold": 8.0,
        "category": "Performance"
    },
    
    # Content Analysis Metrics
    "word_count": {
        "name": "Word Count",
        "description": "Number of words in generated text",
        "range": "0+ words",
        "good_threshold": 50,
        "category": "Content Analysis"
    },
    "sentence_count": {
        "name": "Sentence Count", 
        "description": "Number of sentences in generated text",
        "range": "0+ sentences",
        "good_threshold": 3,
        "category": "Content Analysis"
    },
    "vocabulary_diversity": {
        "name": "Vocabulary Diversity",
        "description": "Ratio of unique words to total words",
        "range": "0-1 (higher is better)",
        "good_threshold": 0.7,
        "category": "Content Analysis"
    },
    "avg_sentence_length": {
        "name": "Avg Sentence Length",
        "description": "Average words per sentence",
        "range": "0+ words",
        "good_threshold": 15,
        "category": "Content Analysis"
    },
    
    # Advanced Quality Metrics
    "perplexity": {
        "name": "Perplexity",
        "description": "Model uncertainty/confidence (lower is better)",
        "range": "1+ (lower is better)",
        "good_threshold": 50,
        "category": "Advanced Quality"
    },
    "coherence_score": {
        "name": "Coherence Score",
        "description": "Logical flow and consistency of text",
        "range": "0-1 (higher is better)",
        "good_threshold": 0.7,
        "category": "Advanced Quality"
    },
    "relevance_score": {
        "name": "Relevance Score",
        "description": "How well response addresses input",
        "range": "0-1 (higher is better)",
        "good_threshold": 0.8,
        "category": "Advanced Quality"
    },
    "fluency_score": {
        "name": "Fluency Score",
        "description": "Grammar and readability quality",
        "range": "0-1 (higher is better)",
        "good_threshold": 0.8,
        "category": "Advanced Quality"
    },
    
    # Cost and Efficiency Metrics
    "cost_per_token": {
        "name": "Cost per Token",
        "description": "Cost efficiency in USD per token",
        "range": "0+ USD (lower is better)",
        "good_threshold": 0.00001,
        "category": "Cost & Efficiency"
    },
    "energy_consumption": {
        "name": "Energy Consumption",
        "description": "Estimated energy usage in Wh",
        "range": "0+ Wh (lower is better)",
        "good_threshold": 1.0,
        "category": "Cost & Efficiency"
    },
    "throughput": {
        "name": "Throughput",
        "description": "Requests processed per minute",
        "range": "0+ req/min (higher is better)",
        "good_threshold": 60,
        "category": "Cost & Efficiency"
    },
    
    # Safety and Ethics Metrics
    "toxicity_score": {
        "name": "Toxicity Score",
        "description": "Harmful content detection (lower is better)",
        "range": "0-1 (lower is better)",
        "good_threshold": 0.1,
        "category": "Safety & Ethics"
    },
    "bias_score": {
        "name": "Bias Score",
        "description": "Detected bias in output (lower is better)",
        "range": "0-1 (lower is better)",
        "good_threshold": 0.2,
        "category": "Safety & Ethics"
    },
    "safety_score": {
        "name": "Safety Score",
        "description": "Overall safety rating (higher is better)",
        "range": "0-1 (higher is better)",
        "good_threshold": 0.9,
        "category": "Safety & Ethics"
    }
}

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
        st.session_state.voice_enabled = False
    if "voice_input_text" not in st.session_state:
        st.session_state.voice_input_text = ""
    if "pending_voice_input" not in st.session_state:
        st.session_state.pending_voice_input = ""
    if "voice_response_enabled" not in st.session_state:
        st.session_state.voice_response_enabled = True
    if "auto_speech" not in st.session_state:
        st.session_state.auto_speech = False
    if "voice_language" not in st.session_state:
        st.session_state.voice_language = "en"
    if "last_audio_response" not in st.session_state:
        st.session_state.last_audio_response = None
    
    # Comparison state
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = {}
    if "selected_metrics" not in st.session_state:
        st.session_state.selected_metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1", "inference_time", "word_count"]

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

def calculate_extended_metrics(result):
    """Calculate extended metrics for comprehensive comparison"""
    output = result.get("output", "")
    inference_time = result.get("inference_time", 0)
    
    if not output:
        return {}
    
    words = output.split()
    sentences = [s.strip() for s in output.split('.') if s.strip()]
    
    extended_metrics = {}
    
    # Basic counts
    extended_metrics["word_count"] = len(words)
    extended_metrics["sentence_count"] = len(sentences)
    extended_metrics["character_count"] = len(output)
    
    # Vocabulary analysis
    unique_words = set(word.lower() for word in words)
    extended_metrics["vocabulary_diversity"] = len(unique_words) / len(words) if words else 0
    extended_metrics["avg_sentence_length"] = len(words) / len(sentences) if sentences else 0
    
    # Performance metrics
    extended_metrics["tokens_per_second"] = len(words) / inference_time if inference_time > 0 else 0
    
    # Estimated metrics (would be calculated by actual models in production)
    extended_metrics["coherence_score"] = min(1.0, 0.5 + (extended_metrics["vocabulary_diversity"] * 0.3))
    extended_metrics["fluency_score"] = min(1.0, 0.6 + (extended_metrics["avg_sentence_length"] / 20))
    extended_metrics["relevance_score"] = 0.8  # Would be calculated by semantic analysis
    
    # Mock cost calculations (replace with actual API costs)
    if "gpt" in result.get("model", "").lower():
        extended_metrics["cost_per_token"] = 0.00003  # GPT-4o pricing
    else:
        extended_metrics["cost_per_token"] = 0.00001  # LLaMA (open source, hosting cost)
    
    extended_metrics["total_cost"] = extended_metrics["cost_per_token"] * len(words)
    
    return extended_metrics

def generate_comprehensive_charts(evaluation_df):
    """Generate comprehensive interactive charts for model comparison"""
    if evaluation_df is None or evaluation_df.empty:
        return None
    
    charts = {}
    
    # Get metric columns by category
    all_metrics = [col for col in evaluation_df.columns 
                  if col not in ["model", "quality_category"]]
    
    metric_categories = {}
    for metric in all_metrics:
        if metric in METRICS_DEFINITIONS:
            category = METRICS_DEFINITIONS[metric]["category"]
            if category not in metric_categories:
                metric_categories[category] = []
            metric_categories[category].append(metric)
        else:
            if "Other" not in metric_categories:
                metric_categories["Other"] = []
            metric_categories["Other"].append(metric)
    
    # 1. Category-wise comparison charts
    for category, metrics in metric_categories.items():
        if len(metrics) > 0:
            fig = go.Figure()
            
            for metric in metrics:
                if metric in evaluation_df.columns:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=evaluation_df["model"],
                        y=evaluation_df[metric],
                        text=evaluation_df[metric].round(3),
                        textposition="auto"
                    ))
            
            fig.update_layout(
                title=f"{category} Metrics Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                template="plotly_white",
                height=500,
                barmode='group'
            )
            
            charts[f"category_{category.lower().replace(' ', '_')}"] = fig
    
    # 2. Radar chart for overall comparison
    if len(all_metrics) >= 3:
        fig_radar = go.Figure()
        
        for idx, row in evaluation_df.iterrows():
            # Normalize values for radar chart (0-1 scale)
            values = []
            labels = []
            
            for metric in all_metrics:
                if metric in row and pd.notna(row[metric]):
                    # Normalize based on metric type
                    if metric in METRICS_DEFINITIONS:
                        if METRICS_DEFINITIONS[metric]["range"].endswith("(lower is better)"):
                            # For "lower is better" metrics, invert the scale
                            max_val = evaluation_df[metric].max()
                            min_val = evaluation_df[metric].min()
                            if max_val != min_val:
                                normalized = 1 - ((row[metric] - min_val) / (max_val - min_val))
                            else:
                                normalized = 0.5
                        else:
                            # For "higher is better" metrics
                            max_val = evaluation_df[metric].max()
                            min_val = evaluation_df[metric].min()
                            if max_val != min_val:
                                normalized = (row[metric] - min_val) / (max_val - min_val)
                            else:
                                normalized = 0.5
                    else:
                        # Default normalization
                        max_val = evaluation_df[metric].max()
                        min_val = evaluation_df[metric].min()
                        if max_val != min_val:
                            normalized = (row[metric] - min_val) / (max_val - min_val)
                        else:
                            normalized = 0.5
                    
                    values.append(normalized)
                    labels.append(metric)
            
            # Close the radar chart
            if values:
                values.append(values[0])
                labels.append(labels[0])
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself',
                    name=row["model"],
                    line=dict(width=3)
                ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="🕸️ Overall Model Performance Radar (Normalized)",
            template="plotly_white",
            height=600
        )
        
        charts["radar_overall"] = fig_radar
    
    # 3. Performance vs Quality scatter plot
    performance_metrics = ["inference_time", "tokens_per_second", "throughput"]
    quality_metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1", "coherence_score"]
    
    perf_metric = None
    qual_metric = None
    
    for metric in performance_metrics:
        if metric in evaluation_df.columns:
            perf_metric = metric
            break
    
    for metric in quality_metrics:
        if metric in evaluation_df.columns:
            qual_metric = metric
            break
    
    if perf_metric and qual_metric:
        fig_scatter = px.scatter(
            evaluation_df,
            x=perf_metric,
            y=qual_metric,
            color="model",
            size="word_count" if "word_count" in evaluation_df.columns else [100]*len(evaluation_df),
            hover_data=all_metrics,
            title=f"⚡ Performance vs Quality: {qual_metric} vs {perf_metric}",
            labels={
                perf_metric: f"{perf_metric.replace('_', ' ').title()}",
                qual_metric: f"{qual_metric.replace('_', ' ').title()}"
            },
            template="plotly_white",
            height=500
        )
        
        charts["scatter_perf_quality"] = fig_scatter
    
    # 4. Cost effectiveness analysis
    if "total_cost" in evaluation_df.columns and qual_metric:
        fig_cost = px.scatter(
            evaluation_df,
            x="total_cost",
            y=qual_metric,
            color="model",
            size="word_count" if "word_count" in evaluation_df.columns else [100]*len(evaluation_df),
            title=f"💰 Cost vs Quality Analysis",
            labels={
                "total_cost": "Total Cost (USD)",
                qual_metric: f"{qual_metric.replace('_', ' ').title()}"
            },
            template="plotly_white",
            height=500
        )
        
        charts["cost_effectiveness"] = fig_cost
    
    # 5. Detailed metrics heatmap
    if len(all_metrics) >= 2 and len(evaluation_df) >= 2:
        # Normalize all metrics for heatmap
        heatmap_data = evaluation_df[all_metrics].copy()
        
        for col in all_metrics:
            if col in heatmap_data.columns:
                col_max = heatmap_data[col].max()
                col_min = heatmap_data[col].min()
                if col_max != col_min:
                    heatmap_data[col] = (heatmap_data[col] - col_min) / (col_max - col_min)
                else:
                    heatmap_data[col] = 0.5
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[col.replace('_', ' ').title() for col in all_metrics],
            y=evaluation_df["model"].tolist(),
            colorscale='RdYlBu_r',
            hoverongaps=False,
            text=heatmap_data.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        
        fig_heatmap.update_layout(
            title="🔥 Comprehensive Performance Heatmap (Normalized 0-1)",
            xaxis_title="Metrics",
            yaxis_title="Models",
            template="plotly_white",
            height=400
        )
        
        charts["heatmap_comprehensive"] = fig_heatmap
    
    return charts

def create_model_comparison_summary(evaluation_df):
    """Create a comprehensive model comparison summary"""
    if evaluation_df is None or evaluation_df.empty:
        return None
    
    summary = {
        "winner_by_category": {},
        "overall_scores": {},
        "detailed_analysis": {}
    }
    
    models = evaluation_df["model"].tolist()
    
    # Calculate category winners
    metric_categories = {}
    for metric in evaluation_df.columns:
        if metric in METRICS_DEFINITIONS:
            category = METRICS_DEFINITIONS[metric]["category"]
            if category not in metric_categories:
                metric_categories[category] = []
            metric_categories[category].append(metric)
    
    for category, metrics in metric_categories.items():
        category_scores = {}
        for model in models:
            model_data = evaluation_df[evaluation_df["model"] == model].iloc[0]
            score = 0
            count = 0
            
            for metric in metrics:
                if metric in evaluation_df.columns and pd.notna(model_data[metric]):
                    metric_score = model_data[metric]
                    
                    # Normalize based on whether higher or lower is better
                    if metric in METRICS_DEFINITIONS:
                        if METRICS_DEFINITIONS[metric]["range"].endswith("(lower is better)"):
                            # Invert score for "lower is better" metrics
                            max_val = evaluation_df[metric].max()
                            min_val = evaluation_df[metric].min()
                            if max_val != min_val:
                                normalized_score = 1 - ((metric_score - min_val) / (max_val - min_val))
                            else:
                                normalized_score = 0.5
                        else:
                            # Normal score for "higher is better" metrics
                            max_val = evaluation_df[metric].max()
                            min_val = evaluation_df[metric].min()
                            if max_val != min_val:
                                normalized_score = (metric_score - min_val) /