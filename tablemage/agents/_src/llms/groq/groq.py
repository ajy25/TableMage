from llama_index.llms.groq import Groq


from ..api_key_utils import find_key


def build_groq(model: str | None = None, temperature: float = 0.1) -> Groq:
    """Builds a Groq object using LlamaIndex.
    If no Groq API key is found in the .env file, raises a ValueError.

    Parameters
    ----------
    model : str
        The model to use, by default "llama-3.3-70b-versatile".
        Must support function calling.

    temperature : float
        The temperature of the model, by default 0.1.

    Returns
    -------
    Groq
        An Groq object from LlamaIndex.
    """
    if model is None:
        model = "llama-3.3-70b-versatile"
    api_key = find_key("groq")
    if not api_key:
        raise ValueError("No Groq API key found in .env file")
    return Groq(model=model, temperature=temperature, api_key=api_key)
