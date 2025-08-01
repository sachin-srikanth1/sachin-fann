"""
Configuration module for SaaS-Swarm platform.

Handles environment variables and settings for OpenAI, email, and other services.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for SaaS-Swarm platform."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Email Configuration
    EMAIL_SMTP_SERVER: str = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    EMAIL_SMTP_PORT: int = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    EMAIL_USERNAME: Optional[str] = os.getenv("EMAIL_USERNAME")
    EMAIL_PASSWORD: Optional[str] = os.getenv("EMAIL_PASSWORD")
    EMAIL_FROM_ADDRESS: Optional[str] = os.getenv("EMAIL_FROM_ADDRESS")
    EMAIL_USE_TLS: bool = os.getenv("EMAIL_USE_TLS", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configuration is present."""
        missing = []
        
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        
        if not cls.EMAIL_USERNAME:
            missing.append("EMAIL_USERNAME")
        
        if not cls.EMAIL_PASSWORD:
            missing.append("EMAIL_PASSWORD")
        
        if not cls.EMAIL_FROM_ADDRESS:
            missing.append("EMAIL_FROM_ADDRESS")
        
        if missing:
            print(f"Missing required environment variables: {', '.join(missing)}")
            print("Please check your .env file and ensure all required variables are set.")
            return False
        
        return True
    
    @classmethod
    def get_openai_config(cls) -> dict:
        """Get OpenAI configuration."""
        return {
            "api_key": cls.OPENAI_API_KEY
        }
    
    @classmethod
    def get_email_config(cls) -> dict:
        """Get email configuration."""
        return {
            "smtp_server": cls.EMAIL_SMTP_SERVER,
            "smtp_port": cls.EMAIL_SMTP_PORT,
            "username": cls.EMAIL_USERNAME,
            "password": cls.EMAIL_PASSWORD,
            "from_address": cls.EMAIL_FROM_ADDRESS,
            "use_tls": cls.EMAIL_USE_TLS
        } 