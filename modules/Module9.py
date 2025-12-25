"""Module 9: Chat with Dataset using Gemini API."""

from __future__ import annotations

import os
import pandas as pd
from dotenv import load_dotenv

# Prefer the new google-genai SDK; fall back to the deprecated google-generativeai
try:  # New SDK (google-genai)
    from google import genai  # type: ignore
    _GENAI_MODE = "new"
except Exception:  # pragma: no cover - keep legacy fallback for safety
    try:
        import google.generativeai as genai  # type: ignore
        _GENAI_MODE = "legacy"
    except Exception:
        genai = None
        _GENAI_MODE = "missing"

# Load environment variables
load_dotenv()

class ChatInterface:
    def __init__(self, api_key: str | None = None):
        # Priority: 1. st.secrets (Streamlit Cloud), 2. .env (local), 3. Explicit arg
        import streamlit as st
        if api_key:
            self.api_key = api_key
        else:
            # Try st.secrets first (for Streamlit Cloud), fallback to .env
            try:
                self.api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                self.api_key = None
            
            if not self.api_key:
                self.api_key = os.getenv("GEMINI_API_KEY")
        
        self.client = None
        self.chat_session = None
        
        if self.api_key and genai:
            try:
                if _GENAI_MODE == "new":
                    self.client = genai.Client(api_key=self.api_key)
                elif _GENAI_MODE == "legacy":
                    genai.configure(api_key=self.api_key)  # type: ignore[attr-defined]
                    # Store model handle so we can start chats later
                    self.client = genai.GenerativeModel("gemini-1.5-flash")
                else:
                    self.client = None
            except Exception as e:
                print(f"Failed to initialize Gemini Client: {e}")

    @property
    def masked_key(self) -> str:
        """Return a masked view of the API key for safe debugging."""
        if not self.api_key:
            return "not set"
        visible_tail = self.api_key[-4:]
        masked = "*" * max(len(self.api_key) - 4, 0)
        return f"{masked}{visible_tail}"

    def is_configured(self) -> bool:
        return self.client is not None

    def start_new_chat(self, df: pd.DataFrame, training_results: str | None = None) -> None:
        """Initialize a new chat session with dataset and training context."""
        if not self.client:
            return

        # Create a context prompt about the dataset
        schema = str(df.dtypes)
        summary = str(df.describe())
        head = str(df.head(5))
        
        training_context = f"\nMODEL TRAINING RESULTS:\n{training_results}" if training_results else ""

        system_prompt = f"""
You are an expert data science assistant. You help students understand their AutoML results.
Here is the context of the dataset and the training results:

DATASET SCHEMA:
{schema}

SUMMARY STATISTICS:
{summary}

DATA PREVIEW:
{head}
{training_context}

Please answer the user's questions based on this specific information. 
If asked which model is best, refer to the training results.
If asked about data patterns, refer to the schema and summary.
Be highly specific, task-oriented, and educational. Avoid generic apologies.
"""
        try:
            if _GENAI_MODE == "new":
                self.chat_session = self.client.chats.create(model="gemini-2.5-flash")
                self.chat_session.send_message(system_prompt)
            elif _GENAI_MODE == "legacy":
                # Legacy SDK uses start_chat on the GenerativeModel instance
                self.chat_session = self.client.start_chat(
                    history=[{"role": "user", "parts": [system_prompt]}]
                )
            else:
                self.chat_session = None
        except Exception as e:
            print(f"Error initializing chat: {e}")
            self.chat_session = None

    def send_message(self, message: str) -> str:
        """Send a message to the chat session and get response."""
        if not self.chat_session:
            return "Chat session not initialized. Please configure API key and load a dataset."
        
        try:
            response = self.chat_session.send_message(message)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini: {e}"
