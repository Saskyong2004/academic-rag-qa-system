import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

client = genai.Client(api_key=API_KEY)


def generate_answer(question, context):
    """
    Generate a grounded answer using Gemini and retrieved document context.
    """

    prompt = f"""
You are an academic question-answering assistant.

Answer the user's question using ONLY the provided document context.

If the answer cannot be found in the context, clearly say:
"I could not find the answer in the provided document."

Do not invent or assume information.

User Question:
{question}

Document Context:
{context}

Provide a clear and concise answer.
"""

    try:
        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            generation_config={
                "thinking_level": "low"
            }
        )

        return interaction.output_text

    except Exception as error:
        error_message = str(error)

        # Handle Gemini API rate-limit errors
        if "429" in error_message or "rate limit" in error_message.lower():
            raise RuntimeError(
                "Gemini API rate limit reached. "
                "Please wait and try again later."
            )

        # Handle other Gemini API errors
        raise RuntimeError(
            f"Gemini API error: {error_message}"
        )