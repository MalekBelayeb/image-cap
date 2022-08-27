import os
import numpy as np

from preprocessing.text import create_tokenizer
from NIC import greedy_inference_model, image_dense_lstm, text_emb_lstm
from evaluate import decoder, beam_search

import base64
from io import BytesIO
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from PIL import Image

def extract_feature_from_image_arr(img_arr):

    x = np.expand_dims(img_arr, axis=0)
    x = preprocess_input(x)

    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    return model.predict(x)


def generate_caption_from_image_greedy(img_arr):
    # Encoder
    img_feature = extract_feature_from_image_arr(img_arr)
    
    # load vocabulary
    tokenizer = create_tokenizer(train_dir, token_dir, start_end = True, use_all=True)

    # set relevent parameters
    vocab_size  = tokenizer.num_words or (len(tokenizer.word_index)+1)
    max_len = 24 # use 24 as maximum sentence's length when training the model
    
    # prepare inference model
    NIC_inference = greedy_inference_model(vocab_size, max_len)
    NIC_inference.load_weights(model_dir, by_name = True, skip_mismatch=True)
    
    # Decoder
    caption = decoder(NIC_inference, tokenizer, img_feature, True)
    
    return caption
    

def generate_caption_greedy(b64_img_str): # from_base64_img_str
    
    # broken as of keras-preprocessing:1.2.0

    #img_data = image.load_img(b64_img_str, target_size=(299, 299))
    img_data = b64_img_str
    #img_data = Image.open(BytesIO(base64.urlsafe_b64decode(b64_img_str)))
    #img_data = b64_img_str
    #img_data = img_data.convert('RGB')
    #img_data = img_data.resize((299,299),Image.NEAREST)

    img_arr = image.img_to_array(img_data)
    
    #generate caption
    caption = generate_caption_from_image_greedy(img_arr)
    return caption

def generate_caption_from_image_beam_search(img_arr, beam_width = 5, alpha = 0.7):
    # Encoder
    img_feature = extract_feature_from_image_arr(img_arr)
    
    # load vocabulary
    tokenizer = create_tokenizer(train_dir, token_dir, start_end = True, use_all=True)

    # set relevent parameters
    vocab_size  = tokenizer.num_words or (len(tokenizer.word_index)+1)
    max_len = 24 # use 24 as maximum sentence's length when training the model
    
    # prepare inference model
    NIC_text_emb_lstm = text_emb_lstm(vocab_size)
    NIC_text_emb_lstm.load_weights(model_dir, by_name = True, skip_mismatch=True)
    NIC_image_dense_lstm = image_dense_lstm()
    NIC_image_dense_lstm.load_weights(model_dir, by_name = True, skip_mismatch=True)
    
    # Decoder
    a0, c0 = NIC_image_dense_lstm.predict([img_feature, np.zeros([1, 512]), np.zeros([1, 512])])
    
    res = beam_search(NIC_text_emb_lstm, a0, c0, tokenizer, beam_width, max_len, alpha)
    best_idx = np.argmax(res['scores'])
    caption = tokenizer.sequences_to_texts([res['routes'][best_idx]])[0]
    
    return caption

def generate_caption_beam_search(b64_img_str): # from_base64_img_str
    
    # broken as of keras-preprocessing:1.2.0
    # img_data = image.load_img(BytesIO(base64.urlsafe_b64decode(b64_img_str)), target_size=(299, 299))

    img_data = b64_img_str
    img_data = img_data.convert('RGB')
    img_data = img_data.resize((299,299),Image.NEAREST)

    img_arr = image.img_to_array(img_data)
    #generate caption
    caption = generate_caption_from_image_beam_search(img_arr)
    
    return caption


# use training token set to create vocabulary
train_dir = r'C:\Users\Malek\Desktop\imgcap-master\imgcap-master\app\datasets\Flickr8k_text\Flickr_8k.trainImages.txt'
token_dir = r'C:\Users\Malek\Desktop\imgcap-master\imgcap-master\app\datasets\Flickr8k_text\Flickr8k.token.txt'
# the current best trained model
model_dir = r'C:\Users\Malek\Desktop\imgcap-master\imgcap-master\app\model\current_best.h5'
