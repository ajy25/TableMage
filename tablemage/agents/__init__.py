"""
TableMage Agents Module
-----------------------
This module, which can be optionally imported as `agents` after \
installing optional dependencies via `pip install .[agents]` \
and calling `tm.use_agents()`, provides a simple interface to \
chat with ChatDA (Chat Data Analyst), TableMage's function-calling \
chatbot.

Classes, Objects, and Functions
-------------------------------
ChatDA_UserInterface : Class that provides a simple user interface to chat with ChatDA.

ChatDA : ChatDA.

options : An instance of _AgentsOptions which dictates LLM settings for ChatDA.

set_key : A function to set the API key for ChatDA.
"""


from .ui.app import ChatDA_UserInterface
from .api.chatda import ChatDA
from ._src.options import options
from ._src.llms.api_key_utils import set_key

__all__ = ["ChatDA_UserInterface", "ChatDA", "options", "set_key"]
