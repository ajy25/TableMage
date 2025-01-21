import sys

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np


sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent))
from tablemage.agents.api import ChatDA
from tablemage.agents._src.options import options

from tablemage.agents._src.io.canvas import (
    CanvasCode,
    CanvasFigure,
    CanvasTable,
    CanvasThought,
)


app = Flask(__name__)
CORS(app)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Global variable to store Mage instance
mage: ChatDA = None


options.set_llm("groq", model_name="llama-3.3-70b-versatile", temperature=0)


def chat(msg: str) -> str:
    """
    Chat function that processes natural language queries on the uploaded dataset.
    """
    global mage
    if mage is None:
        return "No dataset uploaded. Please upload a dataset first."
    else:
        return mage.chat(msg)


def get_analysis():
    """
    Get analysis results from the Mage canvas queue.
    """
    return mage._canvas_queue.get_analysis()


@app.route("/api/health")
def health_check():
    return jsonify({"status": "ok", "message": "Server is running"})


@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    """
    Handle dataset upload and store it for the chat function.
    """
    global mage

    app.logger.info("Received upload request")

    if "file" not in request.files:
        app.logger.error("No file part in request")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        app.logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    app.logger.info(f"Processing file: {file.filename}")

    test_size = request.form.get("test_size", 0.2)
    try:
        test_size = float(test_size)
        if not (0.0 <= test_size <= 1.0):
            app.logger.error(f"Invalid test size: {test_size}")
            raise ValueError("Test size must be between 0.0 and 1.0.")
    except ValueError as e:
        app.logger.error(f"Test size error: {str(e)}")
        return jsonify({"error": str(e)}), 400

    try:
        app.logger.info("Reading CSV file...")
        uploaded_data = pd.read_csv(file)  # Removed index_col=0 to be more flexible

        app.logger.info(f"Data shape: {uploaded_data.shape}")
        app.logger.info("Initializing Mage...")
        mage = ChatDA(uploaded_data, test_size=test_size)

        # Convert preview data to JSON-serializable format
        preview_df = uploaded_data.head()
        preview = preview_df.replace({np.nan: None}).to_dict()

        return (
            jsonify(
                {
                    "message": "Dataset uploaded successfully",
                    "rows": uploaded_data.shape[0],
                    "columns": uploaded_data.columns.tolist(),
                    "preview": preview,
                }
            ),
            200,
        )

    except pd.errors.EmptyDataError:
        app.logger.error("Empty CSV file uploaded")
        return jsonify({"error": "The uploaded CSV file is empty"}), 400
    except pd.errors.ParserError as e:
        app.logger.error(f"CSV parsing error: {str(e)}")
        return (
            jsonify(
                {
                    "error": "Failed to parse CSV file. Please ensure it's properly formatted"
                }
            ),
            400,
        )
    except Exception as e:
        app.logger.error(f"Unexpected error during upload: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """
    Handle chat messages and return responses from Mage.
    """
    if mage is None:
        return (
            jsonify({"error": "No dataset uploaded. Please upload a dataset first."}),
            400,
        )

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    try:
        app.logger.info(f"Received chat message: {data['message']}")
        response = chat(data["message"])
        app.logger.info(f"Generated response: {response}")

        # Get any new analysis items generated from this chat
        analysis_items = get_analysis()

        return (
            jsonify({"response": response, "hasAnalysis": len(analysis_items) > 0}),
            200,
        )

    except Exception as e:
        app.logger.error(f"Error in chat: {str(e)}")
        return jsonify({"error": f"Failed to process message: {str(e)}"}), 500


@app.route("/api/analysis", methods=["GET"])
def get_analysis_history():
    if mage is None:
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
                print(
                    f"Sending figure info to frontend: {path_obj.name}"
                )  # Debug print

                items.append(
                    {
                        "file_name": path_obj.name,
                        "file_type": "figure",
                        "file_path": str(path_obj),
                    }
                )
            elif isinstance(item, CanvasTable):
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
        return jsonify(items)
    except Exception as e:
        app.logger.error(f"Error retrieving analysis history: {str(e)}")
        return jsonify({"error": "Failed to retrieve analysis history"}), 500


@app.route("/api/analysis/file/<filename>", methods=["GET"])
def serve_file(filename):
    if mage is None:
        return (
            jsonify({"error": "No dataset uploaded. Please upload a dataset first."}),
            400,
        )

    analysis_items = get_analysis()
    for item in analysis_items:
        print(f"Requested filename: {filename}")  # Debug print
        if isinstance(item, CanvasFigure) and item._path.name == filename:
            print(f"Checking figure path: {item._path}")  # Debug print
            print(f"Figure path name: {item._path.name}")  # Debug print
            file_path = item._path
            print(f"Found matching file: {file_path}")  # Debug print
            print(f"File exists: {file_path.exists()}")  # Debug print

            if file_path.exists():
                return send_file(file_path)

    return jsonify({"error": f"File '{filename}' not found."}), 404


if __name__ == "__main__":
    app.run(debug=True, port=5005)
