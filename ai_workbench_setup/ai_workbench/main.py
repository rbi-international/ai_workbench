import uvicorn
from src.api import app
import yaml

if __name__ == "__main__":
    with open("config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    uvicorn.run(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"],
        timeout_keep_alive=600,
        workers=1,
        limit_concurrency=10
    )