import cv2
from tensorflow.keras.models import load_model
import numpy as np

model=load_model("my_face_mask.h5")

webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    (rval, im) = webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.3,5) 

    # Draw rectangles around each face
    for (x,y,w,h) in faces:
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        print(result)
              
        if result>0.5:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.rectangle(im,(x,y-40),(x+w,y),(0,0,255),-1)
            cv2.putText(im, 'no_mask', (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        else:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(im,(x,y-40),(x+w,y),(0,255,0),-1)
            cv2.putText(im, 'mask', (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
     
        
    # Show the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()

