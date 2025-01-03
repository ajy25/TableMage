from llama_index.llms.ollama import Ollama


def build_ollama(model: str | None = None, temperature: float = 0.0) -> Ollama:
    """Builds a Ollama object using LlamaIndex.

    Parameters
    ----------
    model : str
        The model to use, by default "lmannix/llama3-groq-tool-8b".
        Must support function calling.

    temperature : float
        The temperature of the model, by default 0.0.

    Returns
    -------
    Ollama
        An Ollama object from LlamaIndex.
    """
    if model is None:
        model = "mannix/llama3-groq-tool-8b"
    return Ollama(model=model, temperature=temperature, request_timeout=60)
