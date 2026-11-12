import os

from flask import Flask, render_template, request

from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


def read_score(field_name):
    """Read a 0-100 score from the form, or return None if it is missing or invalid."""
    try:
        value = float(request.form.get(field_name, ""))
    except ValueError:
        return None
    return value if 0 <= value <= 100 else None


@app.route("/", methods=["GET", "POST"])
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method != "POST":
        return render_template("home.html", form={})

    reading_score = read_score("reading_score")
    writing_score = read_score("writing_score")
    if reading_score is None or writing_score is None:
        return render_template(
            "home.html",
            form=request.form,
            error="Enter reading and writing scores between 0 and 100.",
        ), 400

    data = CustomData(
        gender=request.form.get("gender"),
        race_ethnicity=request.form.get("race_ethnicity"),
        parental_level_of_education=request.form.get("parental_level_of_education"),
        lunch=request.form.get("lunch"),
        test_preparation_course=request.form.get("test_preparation_course"),
        reading_score=reading_score,
        writing_score=writing_score,
    )
    pred_df = data.get_data_as_data_frame()
    logging.info(f"Prediction input: {pred_df.to_dict(orient='records')}")

    prediction = PredictPipeline().predict(pred_df)[0]
    predicted_score = int(round(min(max(prediction, 0), 100)))
    logging.info(f"Predicted math score: {predicted_score}")

    return render_template("home.html", form=request.form, results=predicted_score)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
