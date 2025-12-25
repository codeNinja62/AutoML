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

        # Ensure the environment variable is set so genai.Client() can pick it up
        if self.api_key:
            os.environ["GEMINI_API_KEY"] = self.api_key

        # show whether a key was loaded (do NOT print the key)
        # try:
        #     st.info(f"API key loaded: {bool(self.api_key)}")
        # except Exception:
        #     pass

        self.client: genai.Client | None = None
        self.system_prompt: str | None = None
        self.model = "gemini-2.5-flash"
        self.init_error: str | None = None

        try:
            # Pass the key explicitly to avoid env var race / Cloud secrets issues
            self.client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
        except Exception as e:
            self.init_error = str(e)
            # Surface the error in Streamlit UI (so deployed app shows it)
            try:
                st.error(f"Failed to initialize Gemini client: {e}")
            except Exception:
                # Non-Streamlit context; keep silent but record the error
                pass

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

        # Store the system prompt; we'll use generate_content for each user message
        self.system_prompt = system_prompt

    def send_message(self, message: str) -> str:
        if not self.client or not self.system_prompt:
            return "Chat session not initialized."

        try:
            prompt = self.system_prompt + "\nUSER: " + message
            response = self.client.models.generate_content(
                model=self.model, contents=prompt
            )

            if hasattr(response, "text"):
                return response.text

            # Defensive fallback
            return str(response)

        except Exception as e:
            return f"Error communicating with Gemini: {e}"