"""
conftest.py: Shared pytest setup.

Sets dummy config values so tests never use real API keys and can run
without a .env file (needed for CI later). pytest loads this file before
any test module, so these values exist before `settings` is created.
"""
import os

# Environment variables take priority over .env in pydantic-settings
os.environ["SEC_USER_AGENT"] = "Test User test@example.com"
os.environ["FINNHUB_API_KEY"] = "test-key"