"""
foodllm.py — Food LLM powered by Gemini 3.1 Flash Lite (google-genai)
foodllm suggests recipes for meals based on the ingredients and information the user provides.
"""

import os
import sys
import argparse
import textwrap
from google import genai
from google.genai import types

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL = "gemini-3.1-flash-lite"
WRAP_WIDTH = 80

# ── Helpers ────────────────────────────────────────────────────────────────────
def wrap(text): # formats text to wrap
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=WRAP_WIDTH) or [""])
    return "\n".join(lines)

def divider(char="─", width=WRAP_WIDTH): # divider lines for clean header
    return char * width

def print_header(system_prompt): # print initial header
    print(divider("═"))
    print("  Gemini Food LLM Chat (using google-genai)")
    print(divider("═"))
    print("\n" + divider())
    print("  Commands: 'quit' → exit | 'reset' → clear history")
    print(divider() + "\n")

# ── Core chat loop ─────────────────────────────────────────────────────────────
def run_chat(system_prompt): # runs the chat loop, checking for user input and special commands
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    history = []

    print_header(system_prompt)

    initial = "Hello!"
    while True:
        if initial:
            user_input = initial
            initial = None
        else:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in {"quit", "exit"}:
                print("\nGoodbye!")
                break

            if user_input.lower() == "reset":
                history.clear()
                print("\n[History cleared]\n")
                initial = "Hello!"
                continue

        # ── Send message (streaming) ───────────────────────────────────────
        history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        print("\nFood Bot: ", end="", flush=True)
        full_reply = ""

        try:
            response = client.models.generate_content_stream(
                model=MODEL,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1024,
                )
            )
            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_reply += chunk.text

        except Exception as e:
            print(f"\n[Error: {e}]")
            history.pop()  # remove user message to keep history consistent
            continue

        print("\n")
        history.append(
            types.Content(role="model", parts=[types.Part(text=full_reply)])
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default=None)
    parser.add_argument("-f", "--file", type=str, default=None)
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    system_prompt = """
    You are a food expert that suggests foods based on what ingredients the user has available. 
    Ask questions one at a time and be concise.
    Ask if the user is willing to purchase additional ingredients. If so, ask for any allergies and avoid meals containing those ingredients.
    """
    run_chat(system_prompt)

if __name__ == "__main__":
    main()