"""Shared embedding client for Azure OpenAI / AI Foundry. Reads D:\\RAG-I\\.env.

Foundry endpoints (*.services.ai.azure.com) with API version "1" use Azure's
v1 route: the plain OpenAI client pointed at <endpoint>/openai/v1/, with the
deployment name passed as `model`. (AzureOpenAI + a dated api-version gave 404.)
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

REQUIRED = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


def get_client() -> tuple[OpenAI, str]:
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)}")
    base = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/"
    client = OpenAI(base_url=base, api_key=os.environ["AZURE_OPENAI_API_KEY"])
    return client, os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
