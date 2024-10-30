from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Config(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL_NAME: str
    DATA_DIR: str
    VECTOR_DIR: str
    GPT_4_TEXT_MODEL: str


settings = Config()
