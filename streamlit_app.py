"""
Streamlit Cloud launcher for YouTube Topics Pro Dashboard
"""
import sys
from pathlib import Path

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the dashboard
from yt_topics_pro.dashboard.app import run_dashboard

if __name__ == "__main__":
    run_dashboard()
