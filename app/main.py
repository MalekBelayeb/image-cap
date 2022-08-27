from generate_caption import generate_caption_beam_search, generate_caption_greedy
from flask import Flask, request
import logging
from PIL import Image
import base64
import io
from io import BytesIO

app = Flask(__name__)

@app.route("/")
def error():
    app.logger.info('......................... Someone at "/" {(./.\.)} .........................')
    return "POST at /imgcap/predict/v1 or v2"

@app.route("/imgcap/predict/v1/", methods=['POST'])
def pred_v1():
    app.logger.info('......................... Someone at "../v1/"  {(./.\.)} .........................')
    #req_body = request.get_json(force=True)
    #b64_img_str = req_body['data']
    caption = generate_caption_greedy(Image.open(r"C:\Users\Malek\Desktop\imgcap-master\imgcap-master\app\lion.jpeg").resize((299, 299)))
    caption = caption[0]
    app.logger.info(f'......................... at v1: {caption} .........................')
    return { 'data' : caption }


@app.route("/imgcap/predict/v2/", methods=['POST'])
def pred_v2():

    app.logger.info('......................... Someone at "../v2/"  {(./.\.)} .........................')
    #req_body = request.get_json(force=True)
    #b64_img_str = req_body['data']
    #print()
    caption = generate_caption_beam_search(Image.open(r"C:\Users\Malek\Desktop\imgcap-master\imgcap-master\app\lion.jpeg").resize((299, 299)))
    app.logger.info(f'......................... at v2: {caption} .........................')
    return { 'data' : caption }     


def get_thumbnail(path):
    path = "\\\\?\\"+path # This "\\\\?\\" is used to prevent problems with long Windows paths
    i = Image.open(path)
    return i

caption = generate_caption_beam_search(Image.open(r"C:\Users\Malek\Desktop\imgcap-master\imgcap-master\app\image-attractive.jpg").resize((299, 299)))
print(caption)
def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'



#if __name__ == "__main__":
    # Only for debugging while developing
    #app.run(host='127.0.0.1', debug=True, port=80,threaded=False)
    

if __name__ != "__main__":
    app.logger.setLevel(logging.INFO)

app.logger.info("......................... Serving Started .........................")