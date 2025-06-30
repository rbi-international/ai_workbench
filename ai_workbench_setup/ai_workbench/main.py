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
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config.yaml: {e}")
        sys.exit(1)

def validate_environment():
    """Validate required environment variables"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with the required variables")
        return False
    
    print("✓ Environment variables validated")
    return True

def main():
    print("🚀 Starting AI Workbench...")
    
    # Setup directories
    setup_directories()
    
    # Load configuration
    config = load_config()
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Import app after environment setup
    try:
        from src.api import app
        print("✓ API module imported successfully")
    except ImportError as e:
        print(f"❌ Error importing API module: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    
    # Start the server
    try:
        print(f"🌟 Starting server on {config['api']['host']}:{config['api']['port']}")
        uvicorn.run(
            app,
            host=config["api"]["host"],
            port=config["api"]["port"],
            timeout_keep_alive=config["api"].get("timeout", 600),
            workers=1,
            limit_concurrency=10,
            reload=False  # Disable reload for production
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()