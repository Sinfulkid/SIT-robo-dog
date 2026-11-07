from flask import Flask, render_template

app = Flask(__name__)

ROBOT_STREAM_URL = "http://10.99.201.120:5000/video_feed"

@app.route('/')
def index():
    return render_template('index.html', robot_stream=ROBOT_STREAM_URL)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
