import os
import sys
from pathlib import Path
import streamlit.web.cli as stcli
from dotenv import load_dotenv

def check_environment():
    """Check if all required environment variables are set"""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)

def init_directories():
    """Initialize required directories"""
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/vector_store").mkdir(parents=True, exist_ok=True)

def main():
    """Main entry point of the application"""
    # Load environment variables
    load_dotenv()
    
    # Perform initial checks
    check_environment()
    init_directories()
    
    # Run Streamlit application
    sys.argv = ["streamlit", "run", str(Path("src/ui/app.py"))]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()