import os
from dotenv import load_dotenv

# Set base directory of the app
basedir = os.path.abspath(os.path.dirname(__file__))

# Load the .env and .flaskenv variables
load_dotenv(os.path.join(basedir, "../.env"))


class Config(object):
    """
    Set the config variables for the Flask app

    """
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    
    # JWT Settings
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")  # Change this to a secure secret key
    JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES"))
    
    # Default Admin Credentials
    DEFAULT_USERNAME = os.environ.get("DEFAULT_USERNAME")
    DEFAULT_PASSWORD = os.environ.get("DEFAULT_PASSWORD")