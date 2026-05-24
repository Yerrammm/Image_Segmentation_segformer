import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from model_normal import run_normal_segmentation
from model_medical import run_medical_segmentation

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        model_type = request.form.get("model_type")
        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template("index1.html", error="No file selected")

        filename = secure_filename(file.filename)

        original_path = os.path.join(UPLOAD_FOLDER, filename)
        segmented_path = os.path.join(RESULT_FOLDER, "seg_" + filename)
        masked_path = os.path.join(RESULT_FOLDER, "mask_" + filename)

        file.save(original_path)

        if model_type == "normal":
            accuracy, cm_path = run_normal_segmentation(
                original_path,
                segmented_path,
                masked_path
            )

            return render_template(
                "index1.html",
                original=original_path,
                segmented=segmented_path,
                masked=masked_path,
                accuracy=accuracy,
                cm=cm_path
            )

        elif model_type == "medical":
            message, cm_path = run_medical_segmentation(
                original_path,
                segmented_path,
                masked_path
            )

            return render_template(
                "index1.html",
                original=original_path,
                segmented=segmented_path,
                masked=masked_path,
                message=message,
                cm=cm_path
            )

    return render_template("index1.html")


if __name__ == "__main__":
    app.run(debug=True)