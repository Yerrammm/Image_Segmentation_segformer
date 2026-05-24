import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from flask import Flask, render_template, request
from model_general import run_segmentation

app = Flask(__name__)

# 200 MB upload limit
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        if "image" not in request.files:
            return render_template("index.html", error="No file selected")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", error="Empty filename")

        original_path  = os.path.join(STATIC_FOLDER, "original.png")
        segmented_path = os.path.join(STATIC_FOLDER, "segmented.png")
        masked_path    = os.path.join(STATIC_FOLDER, "masked.png")

        file.save(original_path)

        accuracy = run_segmentation(
            original_path,
            segmented_path,
            masked_path
        )

        return render_template(
            "index.html",
            original="static/original.png",
            segmented="static/segmented.png",
            masked="static/masked.png",
            accuracy=accuracy
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
