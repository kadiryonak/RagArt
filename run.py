#!/usr/bin/env python3
"""
Run script for the Turkish RAG System.

This script provides options to:
1. Start the web interface (Flask server)
2. Run interactive test mode
3. Check system requirements
"""

import os
import sys
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from src.utils import StatusEmoji


def check_requirements() -> bool:
    """
    Check if all required packages are installed.
    
    Returns:
        True if all requirements are met
    """
    required_packages = [
        "flask",
        "flask_cors",
        "langchain",
        "chromadb",
        "sentence_transformers",
        "requests",
        "dotenv"
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            # Try alternate import names
            alt_names = {
                "dotenv": "python-dotenv",
                "flask_cors": "flask-cors",
                "sentence_transformers": "sentence-transformers"
            }
            missing.append(alt_names.get(package, package))
    
    if missing:
        print(f"{StatusEmoji.ERROR} Missing packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    
    print(f"{StatusEmoji.SUCCESS} All required packages installed")
    return True


def check_data_folder() -> bool:
    """
    Check if the data folder exists and contains JSON files.
    
    Returns:
        True if data is available
    """
    data_folder = settings.DATA_FOLDER
    
    if not os.path.exists(data_folder):
        print(f"{StatusEmoji.FOLDER} Data folder not found, creating...")
        os.makedirs(data_folder)
    
    json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
    
    if not json_files:
        print(f"{StatusEmoji.DOCUMENT} No JSON files found, sample data will be created")
        return False
    
    print(f"{StatusEmoji.SUCCESS} Found {len(json_files)} JSON files: {json_files[:5]}...")
    return True


def start_web_server() -> bool:
    """
    Start the Flask web server.
    
    Returns:
        True if server started successfully
    """
    print(f"\n{'='*60}")
    print(f"{StatusEmoji.ROCKET} STARTING TURKISH RAG SYSTEM")
    print(f"{'='*60}")
    
    # Check requirements
    print(f"\n1. Checking packages...")
    if not check_requirements():
        return False
    
    # Check configuration
    print(f"\n2. Configuration status:")
    settings.print_status()
    
    # Check data folder
    print(f"\n3. Checking data folder...")
    check_data_folder()
    
    # Start server
    print(f"\n4. Starting server...")
    print(f"Web interface: http://localhost:{settings.PORT}")
    print(f"\n{StatusEmoji.WARNING} Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print(f"\n\n{StatusEmoji.SUCCESS} Server stopped")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{StatusEmoji.ERROR} Server error: {e}")
        return False
    
    return True


def interactive_mode() -> bool:
    """
    Run the system in interactive test mode.
    
    Returns:
        True if completed successfully
    """
    print(f"\n{'='*60}")
    print(f"{StatusEmoji.CHECK} INTERACTIVE TEST MODE")
    print(f"{'='*60}")
    
    try:
        from src.rag_system import TurkishRAGSystem
        
        # Create system
        api_key = settings.get_api_key()
        model_type = settings.MODEL_TYPE
        
        if model_type in ("deepseek", "openai") and not api_key:
            model_type = "local"
        
        rag_system = TurkishRAGSystem(
            data_folder=settings.DATA_FOLDER,
            model_type=model_type,
            api_key=api_key
        )
        
        print(f"{StatusEmoji.LOADING} Initializing system...")
        rag_system.initialize()
        
        print(f"{StatusEmoji.SUCCESS} System ready! Enter your questions.")
        print(f"{StatusEmoji.INFO} Type 'exit' to quit\n")
        
        while True:
            question = input(f"{StatusEmoji.QUESTION} Question: ").strip()
            
            if question.lower() in ["exit", "quit", "q"]:
                break
            
            if question:
                print(f"{StatusEmoji.LOADING} Processing...")
                result = rag_system.ask(question)
                
                print(f"\n{StatusEmoji.ANSWER} Answer:")
                print(result["answer"])
                print(f"\n{StatusEmoji.SOURCE} Source: {result.get('source', 'unknown')}")
                
                if result.get("source_documents"):
                    print(f"{StatusEmoji.DOCUMENT} Documents used: {len(result['source_documents'])}")
                
                print("-" * 50)
        
        print(f"\n{StatusEmoji.SUCCESS} Interactive mode ended")
        return True
        
    except Exception as e:
        print(f"{StatusEmoji.ERROR} Error: {e}")
        return False


def main():
    """Main entry point."""
    print(f"{StatusEmoji.ROBOT} Turkish RAG System v1.0")
    print("\nOptions:")
    print("1. Start web interface (Flask)")
    print("2. Interactive test mode")
    print("3. System check")
    
    choice = input("\nSelect option (1-3, default: 1): ").strip() or "1"
    
    if choice == "1":
        return start_web_server()
    elif choice == "2":
        return interactive_mode()
    elif choice == "3":
        check_requirements()
        settings.print_status()
        check_data_folder()
        return True
    else:
        print(f"{StatusEmoji.ERROR} Invalid option")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{StatusEmoji.SUCCESS} Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n{StatusEmoji.ERROR} Unexpected error: {e}")
        sys.exit(1)
