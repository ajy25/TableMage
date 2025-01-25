from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from typing import Literal
from pathlib import Path
import sys
import matplotlib

ui_path = Path(__file__).parent.resolve()
path_to_add = str(ui_path.parent.parent.parent)
sys.path.append(path_to_add)


from tablemage.agents.api import ChatDA


from tablemage.agents._src.io.canvas import (
    CanvasCode,
    CanvasFigure,
    CanvasTable,
    CanvasThought,
)

agent: ChatDA = None


chatda_kwargs = {}


def chat(msg: str) -> str:
    """
    Chat function that processes natural language queries on the uploaded dataset.
    """
    global agent
    if agent is None:
        return "No dataset uploaded. Please upload a dataset first."

    else:
        return agent.chat(msg)


def get_analysis():
    return agent._canvas_queue.get_analysis()


# Initialize Flask app
flask_app = Flask(__name__)


@flask_app.route("/")
def index():
    return render_template("index.html")


@flask_app.route("/upload", methods=["POST"])
def upload_dataset():
    """
    Handle dataset upload and store it for the chat function.
    """
    global agent
    global chatda_kwargs
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Get the test size from the form data
    test_size = request.form.get("test_size", 0.2)  # Default to 0.2 if not provided
    try:
        test_size = float(test_size)
        if not (0.0 <= test_size <= 1.0):
            raise ValueError("Test size must be between 0.0 and 1.0.")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        # Read the uploaded CSV file
        uploaded_data = pd.read_csv(file)

        # if the first column is unnamed, drop it
        if uploaded_data.columns[0] == "Unnamed: 0":
            uploaded_data = uploaded_data.drop(columns="Unnamed: 0")

        agent = ChatDA(uploaded_data, test_size=test_size, **chatda_kwargs)

        return jsonify({"message": "Dataset uploaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@flask_app.route("/chat", methods=["POST"])
def chat_route():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    response_message = chat(user_message)
    return jsonify({"response": response_message})


@flask_app.route("/analysis", methods=["GET"])
def get_analysis_history():
    """
    Retrieve the current analysis history (figures, tables, thoughts, code).
    """
    if agent is None:
        return (
            jsonify({"error": "No dataset uploaded. Please upload a dataset first."}),
            400,
        )

    try:
        analysis_items = get_analysis()
        items = []
        for item in analysis_items:
            if isinstance(item, CanvasFigure):
                path_obj = Path(item.path)
                items.append(
                    {
                        "file_name": path_obj.name,
                        "file_type": "figure",
                        "file_path": str(path_obj),
                    }
                )
            elif isinstance(item, CanvasTable):
                # Load the DataFrame and convert to HTML
                path_obj = Path(item.path)
                df = pd.read_pickle(path_obj)
                html_table = df.to_html(classes="table", index=True)
                items.append(
                    {
                        "file_name": path_obj.name,
                        "file_type": "table",
                        "content": html_table,
                    }
                )
            elif isinstance(item, CanvasThought):
                items.append(
                    {
                        "file_type": "thought",
                        "content": item._thought,
                    }
                )
            elif isinstance(item, CanvasCode):
                items.append(
                    {
                        "file_type": "code",
                        "content": item._code,
                    }
                )
            else:
                raise ValueError(f"Unknown item type: {type(item)}")
        return jsonify(items)
    except Exception as e:
        flask_app.logger.error(f"Error retrieving analysis history: {str(e)}")
        return jsonify({"error": "Failed to retrieve analysis history"}), 500


@flask_app.route("/analysis/file/<filename>", methods=["GET"])
def serve_file(filename):
    """
    Serve static files (figures) from the analysis queue.
    """
    if agent is None:
        return (
            jsonify({"error": "No dataset uploaded. Please upload a dataset first."}),
            400,
        )

    analysis_items = get_analysis()
    for item in analysis_items:
        if isinstance(item, CanvasFigure) and item._path.name == filename:
            file_path = item._path
            if file_path.exists():
                return send_file(file_path)

    return jsonify({"error": f"File '{filename}' not found."}), 404


class App:
    def __init__(
        self,
        split_seed: int | None = None,
        system_prompt: str | None = None,
        react: bool | None = None,
        memory_type: Literal["buffer", "vector"] | None = None,
        memory_size: int | None = None,
        tool_rag: bool | None = None,
        tool_rag_top_k: int | None = None,
        python_only: bool | None = None,
        tools_only: bool | None = None,
        multimodal: bool | None = None,
    ):
        """Makes a user interface for the ChatDA agent.

        Parameters
        ----------
        split_seed : int | None
            If None, default seed is used.

        system_prompt : str | None
            If None, default system prompt is used.

        react : bool | None
            If None, default ReAct flag is used. \
            If True, ReAct is used. If False, ReAct is not used.

        memory_type : Literal["buffer", "vector"] | None
            If None, default memory type is used. \
            If "buffer", buffer memory is used. \
            If "vector", vector plus buffer memory is used.

        memory_size : int | None
            If None, default memory size is used.
            The size of the buffer.

        tool_rag : bool | None
            If None, default tool RAG flag is used. \
            If True, tool RAG is used. If False, tool RAG is not used, 
            and all tools are provided to the agent for each query.

        tool_rag_top_k : int | None
            If None, default tool RAG top k is used.
            The number of tools to provide to the agent for each query.

        python_only : bool | None
            If None, default Python-only flag is used. \
            If True, only Python environment is provided. If False, all tools are used.

        tools_only : bool | None
            If None, default tools-only flag is used. \
            If True, only tools are used. If False, all tools are used.

        multimodal : bool | None
            If None, default multimodal flag is used. \
            If True, multimodal model is used for image interpretation.
        """
        matplotlib.use("Agg")

        global chatda_kwargs

        chatda_kwargs = {
            "split_seed": split_seed,
            "system_prompt": system_prompt,
            "react": react,
            "memory_type": memory_type,
            "memory_size": memory_size,
            "tool_rag": tool_rag,
            "tool_rag_top_k": tool_rag_top_k,
            "python_only": python_only,
            "tools_only": tools_only,
            "multimodal": multimodal,
        }
        # remove None key-value pairs
        chatda_kwargs = {k: v for k, v in chatda_kwargs.items() if v is not None}
        self.flask_app = flask_app

    def run(self, host: str = "0.0.0.0", port: str = "5050", debug: bool = False):
        """Runs the Flask app for the ChatDA agent user interface.

        Parameters
        ----------
        host : str
            The host IP address to run the app on.

        port : str
            The port number to run the app on.

        debug : bool
            If True, the app runs in debug mode.
        """
        self.flask_app.run(host=host, debug=debug, port=port)
