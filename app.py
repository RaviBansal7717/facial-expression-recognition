import streamlit as st
from PIL import Image
from model_prediction import load_model,predict_emotion
import numpy as np
import cv2
import os


allowed_image_extensions=["jpg","png","jpeg"]
model_json_file=os.path.abspath("model.json")
model_weights_file=os.path.abspath("model_weights.h5")
model=load_model(model_json_file,model_weights_file)

def layout():
    st.title("Facial Expression Recognition System")
    st.sidebar.title("Facial Expression Recognition System")
    st.sidebar.header("Choose File Type(Image)")
    selection=st.sidebar.selectbox("Choose File Type",["Image"],key="s1")
    st.sidebar.header("Upload File")
    return selection

def run_web_app():
    selection=layout()
    if selection=="Image":
        image_file=st.sidebar.file_uploader("Upload Image",type=allowed_image_extensions,key="f1")
        if image_file is not None:
            image=np.array(Image.open(image_file).convert("RGB"))
            emotion_prediction,boxed_image=predict_emotion(image,model)
            if len(emotion_prediction)!=0:
                emotions=", ".join(result[0] for result in emotion_prediction)
                probabilities=" ".join(str(round(result[1],2)) for result in emotion_prediction)
                st.subheader("Uploaded Image")
                image=cv2.resize(image,(550,370))
                st.image(image)
                st.header("Model's Facial Expression Prediction : ["+emotions+" ]")
                st.subheader("Predicted Emotion Probability : ["+probabilities+" ]")
            else:
                st.warning("Please Upload a different Image as we are unable to detect any faces.")

if __name__=="__main__":
    run_web_app()
