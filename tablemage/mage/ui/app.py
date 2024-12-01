from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd

from pathlib import Path
import sys
import matplotlib

ui_path = Path(__file__).parent.resolve()
path_to_add = str(ui_path.parent.parent.parent)
sys.path.append(path_to_add)


from tablemage.mage.api import Mage

from tablemage.mage._src.io.canvas import (
    CanvasCode,
    CanvasFigure,
    CanvasTable,
    CanvasThought,
)


wizard: Mage = None


def chat(msg: str) -> str:
    """
    Chat function that processes natural language queries on the uploaded dataset.
    """
    global wizard
    if wizard is None:
        return "No dataset uploaded. Please upload a dataset first."

    else:
        return wizard.chat(msg, which="single")


def get_analysis():
    return wizard._canvas_queue.get_analysis()


# Initialize Flask app
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_dataset():
    """
    Handle dataset upload and store it for the chat function.
    """
    global wizard
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
        uploaded_data = pd.read_csv(file, index_col=0)
        wizard = Mage(uploaded_data, test_size=test_size)

        return jsonify({"message": "Dataset uploaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat_route():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    response_message = chat(user_message)
    return jsonify({"response": response_message})


@app.route("/analysis", methods=["GET"])
def get_analysis_history():
    """
    Retrieve the current analysis history (figures, tables, thoughts, code).
    """
    if wizard is None:
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
        app.logger.error(f"Error retrieving analysis history: {str(e)}")
        return jsonify({"error": "Failed to retrieve analysis history"}), 500


@app.route("/analysis/file/<filename>", methods=["GET"])
def serve_file(filename):
    """
    Serve static files (figures) from the analysis queue.
    """
    if wizard is None:
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
    def __init__(self):
        matplotlib.use("Agg")
        self.app = app

    def run(self, debug: bool = False):
        self.app.run(debug=debug)
