from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Config(BaseSettings):
    API_KEY: str
    OPENAI_API_KEY: str
    DATA_PATH: str = "data"
    GPT_4_TEXT_MODEL: str


settings = Config()
