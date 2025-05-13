from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Eureka Settings
    EUREKA_SERVER_URL: str
    EUREKA_APP_NAME: str = "eum-chatbot"
    EUREKA_INSTANCE_HOST: str
    EUREKA_INSTANCE_PORT: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings() 