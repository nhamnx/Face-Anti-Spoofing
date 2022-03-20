from flask import Flask,render_template,Response
from inference import *
import cv2
app = Flask(__name__, template_folder="template")
cam_url = 0

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detection')
def detection():
    return render_template('stream.html')

@app.route('/detection/video')
def video():
    return Response(cam_capture(cam_url), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True, port=8080)

		


