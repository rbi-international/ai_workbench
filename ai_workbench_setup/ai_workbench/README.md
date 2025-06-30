# AI Workbench - Professional AI Platform

A modern, professional AI workbench inspired by ChatGPT and Claude interfaces. This platform provides text summarization, translation, chat capabilities, and document intelligence without voice components for maximum compatibility and performance.

## ✨ Features

- **🤖 Professional Chat Interface** - ChatGPT/Claude-style conversational AI
- **📝 Text Summarization** - Advanced summarization with multiple models
- **🌐 Language Translation** - Multi-language translation capabilities  
- **📄 Document Intelligence** - Upload and query documents (PDF, images, text)
- **📊 Analytics & Monitoring** - Performance metrics and system monitoring
- **🔄 Model Comparison** - Compare outputs from different AI models
- **⚡ Professional UI** - Modern, responsive interface design

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) Hugging Face token for LLaMA models

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ai_workbench
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

4. **Run the application**
```bash
python startup.py
```

This will start both the API server and frontend automatically.

### Manual Startup

If you prefer to start services manually:

1. **Start API Server**
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

2. **Start Frontend** (in a new terminal)
```bash
streamlit run frontend.py --server.port 8501
```

## 🌐 Access Points

- **Frontend**: http://127.0.0.1:8501
- **API**: http://127.0.0.1:8000  
- **API Documentation**: http://127.0.0.1:8000/docs

## 📖 Usage Guide

### Chat Interface

1. Navigate to the **Chat** tab
2. Select your preferred AI model
3. Type your message and press Enter
4. The AI will respond in real-time

### Text Summarization

1. Go to the **Summarization** tab
2. Paste your text in the input area
3. Adjust parameters (temperature, max tokens, etc.)
4. Click "Generate Summary"
5. View results from multiple models with comparison metrics

### Language Translation

1. Select the **Translation** tab
2. Enter text to translate
3. Choose target language
4. Adjust translation parameters
5. Compare translations across models

### Document Intelligence

1. Visit the **Documents** tab
2. Upload PDF, image, or text files
3. Ask questions about your documents
4. Get AI-powered answers based on document content

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

- **Models**: Enable/disable OpenAI or LLaMA models
- **API Settings**: Host, port, timeout configurations
- **RAG Settings**: Vector database and embedding configurations
- **Ethics**: Content filtering and safety thresholds

## 🏗️ Architecture

```
ai_workbench/
├── src/
│   ├── api.py              # FastAPI backend
│   ├── models/             # AI model implementations
│   ├── tasks/              # Task processors (chat, summarization, etc.)
│   ├── rag/                # Document retrieval system
│   └── ...
├── frontend.py             # Streamlit frontend
├── config/
│   └── config.yaml         # Configuration file
├── requirements.txt        # Python dependencies
├── startup.py             # Easy startup script
└── .env                   # Environment variables
```

## 🔧 API Endpoints

- `POST /process` - Main processing endpoint for all tasks
- `POST /upload_documents` - Upload documents for RAG
- `GET /health` - Health check
- `GET /models` - List available models
- `GET /usage_stats` - Usage statistics

## 🛠️ Development

### Adding New Models

1. Create a new model class in `src/models/`
2. Implement the `BaseModel` interface
3. Add model configuration to `config.yaml`
4. Register in the API initialization

### Adding New Tasks

1. Create a task processor in `src/tasks/`
2. Add task logic to the main API endpoint
3. Create frontend interface in `frontend.py`

## 📊 Monitoring & Analytics

The platform includes comprehensive monitoring:

- **System Health**: API status and component availability
- **Usage Statistics**: Request counts, response times, costs
- **Model Performance**: Speed and quality comparisons
- **Error Tracking**: Detailed error logging and reporting

## 🔒 Security & Ethics

- Content filtering for harmful material
- Sentiment and toxicity analysis
- Privacy protection for uploaded documents
- Rate limiting and usage controls

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify your OpenAI API key in `.env`
   - Check key permissions and quota

2. **Model Loading Issues**
   - Ensure sufficient system memory for LLaMA models
   - Check CUDA availability for GPU acceleration

3. **Document Processing Errors**
   - Install pytesseract for image OCR
   - Verify PDF file integrity

4. **Port Conflicts**
   - Change ports in `config.yaml` if needed
   - Check for running services on ports 8000/8501

### Performance Optimization

- Use GPU acceleration for local models
- Enable caching for repeated requests
- Adjust model parameters for speed vs quality balance

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Open an issue on GitHub

## 🚀 Future Enhancements

- Additional model integrations
- Advanced RAG capabilities
- Multi-user support
- Custom model fine-tuning
- Real-time collaboration features

---

**Built with ❤️ for the AI community**