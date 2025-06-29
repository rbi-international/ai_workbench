from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import yaml
from utils.logger import setup_logger
from typing import List, Dict

class RAGRetriever:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.config["rag"]["embedding_model"])
        self.vector_db = Chroma(
            collection_name=self.config["rag"]["collection_name"],
            embedding_function=self.embedding_model,
            persist_directory="./chroma_db"
        )
        self.logger.info("RAGRetriever initialized")

    def add_documents(self, documents: List[str]):
        try:
            self.vector_db.add_texts(documents)
            self.logger.info(f"Added {len(documents)} documents to vector DB")
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        try:
            results = self.vector_db.similarity_search(query, k=k)
            self.logger.info(f"Retrieved {len(results)} documents for query")
            return [{"text": doc.page_content, "metadata": doc.metadata} for doc in results]
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            raise