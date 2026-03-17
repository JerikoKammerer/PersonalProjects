"""
AI Trading Lab - Main Entry Point
Run this file to launch the application.
"""

import sys
import os

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui import main

if __name__ == "__main__":
    main()
