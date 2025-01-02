from dotenv import load_dotenv
from pathlib import Path
import os
from typing import Literal

# clear environment
try:
    os.environ.pop("OPENAI_API")
except KeyError:
    pass
try:
    os.environ.pop("GROQ_API")
except KeyError:
    pass

dotenv_path = Path(__file__).parent.parent.parent.parent.resolve() / ".env"
if not dotenv_path.exists():
    with open(dotenv_path, "w") as f:
        f.write("OPENAI_API_KEY=...\nGROQ_API_KEY=...\n")
    raise Warning(
        "No .env file found. Creating a new .env file. "
        "Please use `tm.agents.set_key()` to set your API keys. "
        "You only need to do this once."
    )
else:
    if not load_dotenv(dotenv_path=str(dotenv_path)):
        raise RuntimeError("Error loading .env file.")


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
    if llm_type == "openai":
        api_key = (
            str(os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        )
        if api_key == "..." or api_key is None:
            raise ValueError("OpenAI API key not found in .env file.")
    elif llm_type == "groq":
        api_key = str(os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None
        if api_key == "..." or api_key is None:
            raise ValueError("Groq API key not found in .env file.")
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

    # reload the .env file
    if not load_dotenv(dotenv_path=str(dotenv_path)):
        raise RuntimeError("Error loading .env file.")
