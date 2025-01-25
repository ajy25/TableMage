from pathlib import Path
from llama_index.core.schema import ImageDocument
from llama_index.core.multi_modal_llms import MultiModalLLM


def describe_image(
    multimodal_model: MultiModalLLM, image_path: Path, text_description: str
) -> str:
    """Describes an image using the multimodal model.

    Parameters
    ----------
    multimodal_model : MultiModalLLM
        The multimodal model to use for image description.

    image_path : Path
        Path to the image.

    text_description : str
        Non AI generated supplementary text description to add to the image description.

    Returns
    -------
    str
        Description of the image.
    """
    image_document = ImageDocument(image_path=str(image_path))

    prompt = "Describe the statistical figure in detail. Be specific. "
    "Focus only on insights you can infer from the figure, "
    "rather than color, size, or other non-data related aspects.\n\n"
    "Here is some additional contextual information about the figure:\n\n"
    f"'{text_description}'"

    str_description = multimodal_model.complete(
        prompt=prompt, image_documents=[image_document]
    ).text
    return str_description
