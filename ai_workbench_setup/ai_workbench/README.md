AI Workbench 🚀
A modular, production-grade AI platform for comparing LLaMA 3.1 and GPT-4o across tasks like summarization, translation, and chat. Packed with features like RAG, fine-tuning, voice I/O, model fusion, collaboration, ethics analysis, and cost control, this project is your go-to for experimenting with advanced AI workflows.


📚 Table of Contents

Credits
Features
Setup
Usage
Cost Control
Notes


🎓 Credits
This project was inspired by and built upon knowledge gained from the course "Generative AI with AI Agents: MCP for Developers" by DSwithBappy on Udemy. A huge thanks to DSwithBappy for providing invaluable insights and learning resources that significantly contributed to the development of this project.

🌟 Features

Click to expand features


Tasks: Perform summarization, translation, and conversational chat with ease.
Retrieval-Augmented Generation (RAG): Powered by Chroma vector database for context-aware responses.
Fine-Tuning: LoRA-based fine-tuning for LLaMA models to customize performance.
Playground: Interactive sliders for tweaking model parameters (temperature, top-p, max_tokens).
Metrics: Evaluate outputs with ROUGE, BERTScore, BLEU, METEOR, and perplexity.
Visualizations: Dynamic bar charts and radar plots for performance comparison.
AI Tutor: Explains model performance with detailed insights.
Crowdsourcing: Submit and moderate user-contributed datasets.
Voice I/O: Speech-to-text and text-to-speech capabilities for seamless interaction.
Model Fusion: Combine outputs from multiple models for enhanced results.
Collaboration: Real-time WebSocket-based arena for model battles.
Ethics: Sentiment and toxicity analysis to ensure responsible AI usage.
Cost Control: Word limit enforcement (100 words/output) and API caching to minimize costs.




🛠️ Setup
Get started with AI Workbench in just a few steps!

1. Install Dependencies

Run the following command to install required Python packages:
pip install -r requirements.txt

Ensure you’re using Python 3.11.13 for compatibility with all libraries.



2. Set Up Environment


Add OpenAI API Key: Edit ai_workbench/.env and replace your_openai_api_key_here with your actual OpenAI API key.OPENAI_API_KEY=your_actual_openai_api_key


Install Tesseract OCR: Download from GitHub and add to PATH (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe).
Install PyAudio: Use pip install pyaudio. If errors occur, install portaudio via Chocolatey:choco install portaudio


LLaMA Access: Ensure access to LLaMA 3.1 via Hugging Face (update config.yaml if using a different variant).


3. Run FastAPI Server

Start the backend API server:
python main.py

This runs the FastAPI server on http://0.0.0.0:8000.



4. Run Streamlit Frontend

Launch the interactive frontend:
streamlit run frontend.py

Access the UI at http://localhost:8501.



5. Run Tests

Validate the setup with unit tests:
pytest tests/




🚀 Usage

Open http://localhost:8501 in your browser.
Select a task (summarization, translation, or chat).
Adjust model parameters (temperature, top-p, max_tokens) via sliders.
Upload documents or audio for RAG or voice input.
Process tasks and explore outputs, metrics, visualizations, and AI Tutor explanations.
Use the collaboration arena for real-time model battles.
Monitor ethical AI metrics and costs via the dashboard.


💰 Cost Control
Keep your API usage in check with these features:

Word Limit: Outputs capped at 100 words per response (configurable in config.yaml).
API Caching: OpenAI responses cached in data/cache/ to reduce redundant calls.
Usage Tracking: Monitor token and word usage in logs/cost_tracker.log.
Estimated Cost: Approximately $0.50–$1.50 per 1000 API calls (based on GPT-4o pricing).


📝 Notes

GPU Recommended: Use a GPU for faster LLaMA inference.
Voice Input: Requires a microphone and WAV files for speech-to-text.
WebSocket Arena: Needs a JavaScript client for full interactivity.
Cost Monitoring: Regularly check logs/cost_tracker.log to manage OpenAI API expenses.


Feel free to contribute, report issues, or suggest enhancements via GitHub Issues!