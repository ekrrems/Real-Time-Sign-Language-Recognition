import cv2
import numpy as np
import pickle

def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img/255
    
    return img

cap = cv2.VideoCapture(0)
cap.set(3,480)
cap.set(4,480)

pickle_in = open("sign_model_normalized.p","rb")
model = pickle.load(pickle_in)



while 1:
    succ , frame = cap.read()
    img = np.array(frame[100:400,100:400])
    cv2.rectangle(frame,(100,100),(400,400),(0,255,0),1)
    img = cv2.resize(img,(28,28))
    img = preprocess(img)
    
    img = img.reshape(1,28,28,1)
    
    
    prediction = model.predict(img)
    print(str(np.argmax(prediction.astype('int'))))
    
    
    cv2.putText(frame,str(np.argmax(prediction.astype('int'))),(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break