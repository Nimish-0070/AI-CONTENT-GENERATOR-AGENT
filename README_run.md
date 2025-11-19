# Run checklist

1. Create virtualenv and activate (see earlier).
2. pip install -r requirements.txt
3. Copy .env.example -> .env and add:
   - GEMINI_API_KEY: get it from Google AI Studio (API Keys).
   - TAVILY_API_KEY: from Tavily dashboard (optional).
4. Run: streamlit run app.py
5. If you get an error "ModuleNotFoundError: No module named 'google'",
   run: pip install google-genai
6. If upgrading gemini client is necessary:
   pip install -U google-genai
