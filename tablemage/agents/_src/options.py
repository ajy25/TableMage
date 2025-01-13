from typing import Literal, Callable
from functools import partial
import os
from ._debug.logger import print_debug
from .llms.groq.groq import build_groq
from .llms.openai.openai import build_openai, build_openai_multimodal
from .llms.ollama.ollama import build_ollama
from .llms.api_key_utils import key_exists


class _AgentsOptions:
    def __init__(self):
        self._multimodal_llm_build_function = None
        self._llm_build_function = None

        self._cpu_count = os.cpu_count()
        if self._cpu_count is None:
            self._cpu_count = 1

        if key_exists("openai"):
            self._llm_build_function = build_openai
            try:
                self._multimodal_llm_build_function = build_openai_multimodal
            except Exception as e:
                print_debug("Multimodal LLM build function not set: " + str(e))
        elif key_exists("groq"):
            self._llm_build_function = build_groq
        else:
            self._llm_build_function = build_ollama

        self._multimodal = (
            True if self._multimodal_llm_build_function is not None else False
        )
        print_debug(
            "A new _AgentOptions object has been created "
            "with LLM build function: "
            + str(self._llm_build_function)
            + " and multimodal LLM build function: "
            + str(self._multimodal_llm_build_function)
        )

    def set_llm(
        self,
        llm_type: Literal["openai", "groq", "ollama"],
        model_name: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        """Sets the LLM type.

        Parameters
        ----------
        llm_type : Literal["openai", "groq", "ollama"]
            The type of LLM to use.

        model_name : str, optional
            The name of the model to use, by default None.
            If None, the default model for llm_type will be used.

        temperature : float, optional
            The temperature to use for the LLM, by default 0.0.
        """
        if llm_type == "openai":
            if not key_exists("openai"):
                raise ValueError("OpenAI API key not found in .env file.")
            self._llm_build_function = partial(
                build_openai, temperature=temperature, model=model_name
            )
            print_debug(
                "OpenAI LLM build function has been set to: "
                + str(self._llm_build_function)
            )
        elif llm_type == "groq":
            if not key_exists("groq"):
                raise ValueError("GROQ API key not found in .env file.")
            self._llm_build_function = partial(
                build_groq, temperature=temperature, model=model_name
            )
            print_debug(
                "GROQ LLM build function has been set to: "
                + str(self._llm_build_function)
            )
        elif llm_type == "ollama":
            self._llm_build_function = partial(
                build_ollama, temperature=temperature, model=model_name
            )
        else:
            raise ValueError("Invalid LLM type specified.")

    def set_multimodal_llm(self, llm_type: Literal["openai"]) -> None:
        """Sets the multimodal LLM type.

        Parameters
        ----------
        llm_type : Literal["openai"]
            The type of multimodal LLM to use.
        """
        if llm_type == "openai":
            if not key_exists("openai"):
                raise ValueError("OpenAI API key not found in .env file.")
            self._multimodal_llm_build_function = build_openai_multimodal
        else:
            raise ValueError("Invalid multimodal LLM type specified.")

    @property
    def llm_build_function(self) -> Callable:
        """The function to build the LLM."""
        return self._llm_build_function

    @property
    def multimodal_llm_build_function(self) -> Callable:
        """The function to build the multimodal LLM."""
        return self._multimodal_llm_build_function

    @property
    def multimodal(self) -> bool:
        """Whether the LLM is multimodal."""
        return self._multimodal


options = _AgentsOptions()
