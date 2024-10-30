from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    OPENAI_API_KEY: str
    DATA_PATH: str
    GPT_4_TEXT_MODEL: str
    API_KEY: str
settings = Config()

