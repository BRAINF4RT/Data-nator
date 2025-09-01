import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
DEFAULT_NUM_RESULTS = 5

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")
