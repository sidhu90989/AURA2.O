from dotenv import load_dotenv
import os

class Config:
    def __init__(self):
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_KEY")
        self.neo4j_pass = os.getenv("NEO4J_PASSWORD")