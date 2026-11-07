import os
from flask import Flask, render_template

# Point template_folder to the sibling 'templates' directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates'))
app = Flask(__name__, template_folder=template_dir)

ROBOT_STREAM_URL = "http://192.168.123.18:5000/video_feed"

@app.route('/')
def index():
    return render_template('index.html', robot_stream=ROBOT_STREAM_URL)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
