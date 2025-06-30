import uvicorn
import yaml
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "data",
        "data/cache", 
        "data/crowdsourced",
        "logs",
        "chroma_db"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory created/verified: {directory}")

def load_config():
    """Load configuration with error handling"""
    try:
        with open("config/config.yaml", 'r') as file:
            config = yaml.safe_load(file)
        print("✓ Configuration loaded successfully")
        return config
    except FileNotFoundError:
        print("❌ Error: config/config.yaml not found")
        print("Please ensure the config file exists in the config/ directory")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config.yaml: {e}")
        sys.exit(1)

def validate_environment():
    """Validate required environment variables"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.strip() == "":
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease check your .env file and ensure it contains:")
        print("OPENAI_API_KEY=\"your_actual_api_key_here\"")
        print("HUGGINGFACE_TOKEN=\"your_actual_token_here\"")
        return False
    
    print("✓ Environment variables validated")
    
    # Verify API key format (basic check)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and not api_key.startswith("sk-"):
        print("⚠️  Warning: OpenAI API key should start with 'sk-'")
    
    return True

def main():
    print("🚀 Starting AI Workbench...")
    print("=" * 60)
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Load configuration
    print("\n⚙️  Loading configuration...")
    config = load_config()
    
    # Validate environment
    print("\n🔐 Validating environment...")
    if not validate_environment():
        sys.exit(1)
    
    # Import app after environment setup
    print("\n📦 Loading application modules...")
    try:
        from src.api import app
        print("✓ API module imported successfully")
    except ImportError as e:
        print(f"❌ Error importing API module: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Ensure you're in the correct directory")
        print("3. Check that all source files are present")
        sys.exit(1)
    
    # Get configuration values
    host = config["api"]["host"]
    port = config["api"]["port"]
    
    # Start the server
    print(f"\n🌟 Starting AI Workbench server...")
    print(f"   🔗 API Server: http://{host}:{port}")
    print(f"   🖥️  Streamlit UI: http://localhost:8501 (run separately)")
    print(f"   📚 API Docs: http://{host}:{port}/docs")
    print("=" * 60)
    print("🎯 To start the UI, run in another terminal:")
    print("   streamlit run frontend.py")
    print("=" * 60)
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            timeout_keep_alive=config["api"].get("timeout", 600),
            workers=1,
            limit_concurrency=10,
            reload=False,  # Disable reload for production
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\n🔧 Common solutions:")
        print(f"1. Check if port {port} is already in use")
        print("2. Try running with different port: uvicorn src.api:app --port 8001")
        print("3. Check firewall settings")
        sys.exit(1)

if __name__ == "__main__":
    main()