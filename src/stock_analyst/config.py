"""
config.py: Central app settings.

Loads configuration (API keys, SEC user agent) from .env so secrets never
live in code. Import `settings` wherever a config value is needed.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Read from .env; ignore unknown keys so adding new ones never breaks startup
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SEC requires every request to identify the caller as "Name email"
    sec_user_agent: str

    # Finnhub API key for company news (free tier)
    finnhub_api_key: str
    
    # LLM: provider and model come from .env so switching needs no code change
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.5-flash"
    google_api_key: str | None = None   # required only when llm_provider is "gemini"


# One shared instance, created once when the module is first imported
settings = Settings()