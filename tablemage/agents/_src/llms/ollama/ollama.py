from llama_index.llms.ollama import Ollama


def build_ollama(model: str | None = None, temperature: float = 0.0) -> Ollama:
    """Builds an Ollama object using LlamaIndex.

    Parameters
    ----------
    model : str
        The model to use, by default "llama3.1:8b-instruct-q4_K_M".
        Must support function calling.

    temperature : float
        The temperature of the model, by default 0.0.

    Returns
    -------
    Ollama
        An Ollama object from LlamaIndex.
    """
    if model is None:
        model = "llama3.1:8b-instruct-q4_K_M"
    return Ollama(model=model, temperature=temperature, request_timeout=120)
