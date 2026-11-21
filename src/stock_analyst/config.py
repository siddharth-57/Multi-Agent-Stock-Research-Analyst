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


# One shared instance, created once when the module is first imported
settings = Settings()