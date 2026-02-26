#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick connectivity test for the OpenRouter API.
Run this first to verify your API key and model are working.

Usage:
    python test_api.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

API_KEY  = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

if not API_KEY:
    raise SystemExit("❌  OPENAI_API_KEY not found — copy .env.example to .env and add your key.")

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

response = client.chat.completions.create(
    model="deepseek/deepseek-r1-0528-qwen3-8b:free",
    messages=[
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "Who won the World Series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user",      "content": "Where was it played?"},
    ],
    stream=False,
)

print("✅  API connection successful!")
print("Reply:", response.choices[0].message.content)
