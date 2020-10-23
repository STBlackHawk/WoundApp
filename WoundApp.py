from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
import cv2
from flask import Flask, request, render_template
import imutils
import io
import numpy as np
import os
import pickle
from PIL import Image
import subprocess



app = Flask(__name__)


def detect():
		p = subprocess.Popen(['python3', 'yolov5/detect.py', '--weights',\
			'best2.pt', '--conf', '0.1', '--source','test1.jpeg'])
		p.communicate()

		return ('Saved')

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file'

        # process data
        infile = request.files['file']
        print('This is the file name'+infile.filename)
        im = Image.open(io.BytesIO(infile.stream.read()))
        im = im.resize([600, 600])


        #do the type checking here
        im_rgb = np.array(im)[:, :, [2, 1, 0]]
        Image.fromarray(im_rgb).save('test1.jpeg')
        #im.save('test1.jpeg')

        #run the deteciton
        t = detect()
        if t=='Saved':
        	im = Image.open('inference/output/test1.jpeg')

        #read the results
        

    
        H, W = im.size[0], im.size[1]
        X = np.array(
            im.getdata()).reshape(
            W, H, 3).astype(np.uint8)
        
        print(X.shape)

        W = X.shape[0]
        H = X.shape[1]

        img = np.empty((W, H), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape((W, H, 4))

        view[:, :, :3] = X[::-1, :, ::-1]
        view[:, :, 3] = 255

        p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
        p.x_range.range_padding = p.y_range.range_padding = 0

        # must give a vector of images
        p.image_rgba(image=[img], x=0, y=0, dw=10, dh=10)

        # grab the static resources
        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        # render template
        script, div = components(p)
        html = render_template(
            'bokeh.html',
            plot_script=script,
            plot_div=div,
            js_resources=js_resources,
            css_resources=css_resources,
        )
        return html
            
    else: 
        return "Upload"

if __name__ == '__main__':
    app.run()
