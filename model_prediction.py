from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
from mtcnn import MTCNN

EMOTIONS=["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
detector=MTCNN()


def load_model(model_json_file,model_weights_file):
  with open(model_json_file,"r") as json_file:
    model_json=json_file.read()
    model=model_from_json(model_json)

  model.load_weights(model_weights_file)
  return model

def preprocess(cropped_face):
  cropped_face=cv2.cvtColor(cropped_face,cv2.COLOR_RGB2GRAY)
  cropped_face=cv2.resize(cropped_face,(48,48))
  cropped_face=np.expand_dims(cropped_face,axis=(0,-1))/255.
  return cropped_face


def detect_faces(image):
    detected_faces=detector.detect_faces(image)
    cropped_faces=[]
    boxed_image=np.copy(image)
    if len(detected_faces)!=0:
      for face in detected_faces:
          x,y,w,h=face["box"] 
          boxed_image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
          cropped_faces.append(image[y:y+h,x:x+w])
    return detected_faces,cropped_faces,boxed_image

def raise_flag(detected_faces,cropped_faces):
  flag=False
  if len(detected_faces)==0:
    flag=True
  else:
    for cropped_face in cropped_faces:
      if 0 in cropped_face.shape:
        flag=True
  return flag

def predict_emotion(image,model):
    flag=False
    emotion_prediction=[]
    detected_faces,cropped_faces,boxed_image=detect_faces(image)
    flag=raise_flag(detected_faces,cropped_faces)
    if flag==False:  
      for cropped_face in cropped_faces:
          cropped_face=preprocess(cropped_face)
          predictions=model.predict(cropped_face)
          emotion_prediction.append((EMOTIONS[np.argmax(predictions)],np.max(predictions)))
      return (emotion_prediction,boxed_image)
    else:
      return ([],[])

