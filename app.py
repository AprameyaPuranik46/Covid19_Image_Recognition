from os.path import join

import numpy as np
from cv2 import INTER_AREA, imread, resize
from flask import Flask, jsonify, redirect, render_template, request, url_for
from keras import Model
from keras.models import model_from_json
from werkzeug.datastructures import FileStorage
from werkzeug.wrappers.response import Response

app: Flask = Flask(__name__)
UPLOAD_FOLDER: str = "static"

# Load the pre-trained model
with open("ResNET50.json", "r") as json_file:
    model_json: str = json_file.read()
model: Model = model_from_json(model_json)
model.load_weights("ResNET50.keras")

# Define categories for classification
categories: list[str] = ["COVID", "Non-COVID"]


@app.route("/")
def home() -> str:
    """Render the home page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    """Handle image upload and prediction."""
    try:
        file: FileStorage = request.files["file"]
        if file is None or file.filename == "":
            raise ValueError("No selected file")
        file_path = join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        image = np.expand_dims(
            resize(
                imread(file_path).copy(), (250, 250), interpolation=INTER_AREA
            ).astype("float")
            / 255.0,
            axis=0,
        )

        # Make the prediction using the loaded model
        prediction = model.predict(image)
        result: str = categories[np.argmax(prediction)]
        confidence_rates = np.fromstring(str(prediction).strip("[]"), sep=" ")
        covid_confidence = f"{confidence_rates[0] * 100.0:.3f}%"
        non_covid_confidence = f"{confidence_rates[1] * 100.0:.3f}%"

        # Redirect to the results page with the prediction data
        return redirect(
            url_for(
                "results",
                filename=file.filename,
                result=result,
                covid_confidence=covid_confidence,
                non_covid_confidence=non_covid_confidence,
            )
        )

    except KeyError:
        data = jsonify({"error": "No file part"})
        data.status_code = 400
        return data

    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        data = jsonify({"error": "No file part"})
        data.status_code = 500
        return data


@app.route("/results/<filename>")
def results(filename) -> str:
    """Render the results page with the prediction data."""
    result = request.args.get("result")
    covid_confidence = request.args.get("covid_confidence")
    non_covid_confidence = request.args.get("non_covid_confidence")
    return render_template(
        "results.html",
        filename=filename,
        result=result,
        covid_confidence=covid_confidence,
        non_covid_confidence=non_covid_confidence,
    )


if __name__ == "__main__":
    app.run(debug=True)