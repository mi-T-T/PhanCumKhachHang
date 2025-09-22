from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load mô hình và scaler đã huấn luyện
model = joblib.load("kmeans.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    cluster = None
    income = ""
    score = ""

    if request.method == "POST":
        try:
            income = float(request.form["income"])
            score = float(request.form["score"])
            input_scaled = scaler.transform([[income, score]])
            cluster = int(model.predict(input_scaled)[0])
        except Exception as e:
            cluster = f"Lỗi: {e}"

    return render_template("index.html", cluster=cluster, income=income, score=score)

if __name__ == "__main__":
    app.run(debug=True)
