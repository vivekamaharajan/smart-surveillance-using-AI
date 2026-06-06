from flask import Flask, render_template, Response
from detection import generate_frames
import csv

app = Flask(__name__)

LOG_FILE = "incident_logs.csv"


@app.route('/')
def index():

    logs = []

    try:
        with open(LOG_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader)   # skip header
            logs = list(reader)[-10:]   # last 10 logs
    except Exception as e:
        print(e)
        logs = []

    return render_template("index.html", logs=logs)


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)