"""Module 9: Chat with Dataset using Gemini API (google-genai SDK only)."""

from __future__ import annotations

import os
import pandas as pd
from dotenv import load_dotenv

from google import genai

# Load environment variables
load_dotenv()


class ChatInterface:
    def __init__(self, api_key: str | None = None):
        import streamlit as st

        # Priority: explicit arg > st.secrets > .env
        if api_key:
            self.api_key = api_key
        else:
            try:
                self.api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                self.api_key = None

            if not self.api_key:
                self.api_key = os.getenv("GEMINI_API_KEY")

        self.client: genai.Client | None = None
        self.chat_session = None

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize Gemini client: {e}")

    @property
    def masked_key(self) -> str:
        if not self.api_key:
            return "not set"
        return "*" * (len(self.api_key) - 4) + self.api_key[-4:]

    def is_configured(self) -> bool:
        return self.client is not None

    def start_new_chat(
        self,
        df: pd.DataFrame,
        training_results: str | None = None,
    ) -> None:
        if not self.client:
            return

        schema = str(df.dtypes)
        summary = str(df.describe(include="all"))
        head = str(df.head(5))

        training_context = (
            f"\nMODEL TRAINING RESULTS:\n{training_results}"
            if training_results
            else ""
        )

        system_prompt = f"""
You are an expert data science assistant helping students understand AutoML results.

DATASET SCHEMA:
{schema}

SUMMARY STATISTICS:
{summary}

DATA PREVIEW:
{head}
{training_context}

Rules:
- Base answers strictly on the provided dataset and training results
- If asked which model is best, use the training results
- Be specific, technical, and educational
"""

        try:
            self.chat_session = self.client.chats.create(
                model="gemini-2.5-flash"
            )
            self.chat_session.send_message(system_prompt)

        except Exception as e:
            print(f"Error initializing chat: {e}")
            self.chat_session = None

    def send_message(self, message: str) -> str:
        if not self.chat_session:
            return "Chat session not initialized."

        try:
            response = self.chat_session.send_message(message)

            # Most common case
            if hasattr(response, "text"):
                return response.text

            # Defensive fallback for structured responses
            return response.candidates[0].content.parts[0].text

        except Exception as e:
            return f"Error communicating with Gemini: {e}"
