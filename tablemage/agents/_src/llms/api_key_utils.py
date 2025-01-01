from dotenv import load_dotenv
from pathlib import Path
import os
from typing import Literal


dotenv_path = Path(__file__).parent.parent.parent.parent.parent.resolve() / ".env"
if not dotenv_path.exists():
    with open(dotenv_path, "w") as f:
        f.write("OPENAI_API_KEY=...\nGROQ_API_KEY=...\n")


def key_exists(
    llm_type: Literal[
        "openai",
        "groq",
    ]
) -> bool:
    """Reads the .env file and returns whether the API key for the specified LLM type exists.

    Parameters
    ----------
    llm_type : Literal["openai"]
        The type of LLM for which to find the API key.
    """
    load_dotenv(dotenv_path=dotenv_path)

    if llm_type == "openai":
        api_key = (
            str(os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        )
        if api_key == "..." or api_key is None:
            return False
    elif llm_type == "groq":
        api_key = str(os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None
        if api_key == "..." or api_key is None:
            return False
    else:
        raise ValueError("Invalid LLM type specified.")
    return True


def find_key(llm_type: Literal["openai", "groq"]) -> str:
    """Reads the .env file and returns the API key for the specified LLM type.
    If the API key is not found, raises a ValueError.

    Parameters
    ----------
    llm_type : Literal["openai", "groq"]
        The type of LLM for which to find the API key.
    """
    load_dotenv(dotenv_path=dotenv_path)

    if llm_type == "openai":
        api_key = (
            str(os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        )
        if api_key == "..." or api_key is None:
            raise ValueError("OpenAI API key not found in .env file.")
    elif llm_type == "groq":
        api_key = str(os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None
        if api_key == "..." or api_key is None:
            raise ValueError("GROQ API key not found in .env file.")
    else:
        raise ValueError("Invalid LLM type specified.")

    return api_key


def set_key(llm_type: Literal["openai", "groq"], api_key: str) -> None:
    """Writes the specified API key to the .env file.

    Parameters
    ----------
    llm_type : Literal["openai", "groq"]
        The type of LLM for which to set the API key.

    api_key : str
        The API key to set.
    """
    if llm_type == "openai":
        key_name = "OPENAI_API_KEY"
    elif llm_type == "groq":
        key_name = "GROQ_API_KEY"

    with open(dotenv_path, "r") as f:
        lines = f.readlines()

    with open(dotenv_path, "w") as f:
        for line in lines:
            if line.startswith(key_name):
                f.write(f"{key_name}={api_key}\n")
            else:
                f.write(line)
