from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


from ..api_key_utils import find_key


def build_openai(model: str | None = None, temperature: float = 0.1) -> OpenAI:
    """Builds a OpenAI object using LlamaIndex.
    If no OpenAI API key is found in the .env file, raises a ValueError.

    Parameters
    ----------
    model : str
        The model to use, by default "gpt-4o-mini".

    temperature : float
        The temperature of the model, by default 0.1.

    Returns
    -------
    OpenAI
        An OpenAI object from LlamaIndex.
    """
    if model is None:
        model = "gpt-4o-mini"
    api_key = find_key("openai")
    if not api_key:
        raise ValueError("No OpenAI API key found in .env file")
    return OpenAI(model=model, temperature=temperature, api_key=api_key)


def build_openai_multimodal(
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> OpenAIMultiModal:
    api_key = find_key("openai")
    if not api_key:
        raise ValueError("No OpenAI API key found in .env file")

    return OpenAIMultiModal(
        model=model, temperature=temperature, api_key=api_key, max_new_tokens=1500
    )
