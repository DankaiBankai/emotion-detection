
import cv2
import tensorflow as tf
import keras as ks
import numpy as np

#List of emotions
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral') 

#Detection loop
def detect(gray_scale, frame, face_cascade, model):
    #Detect faces in gray scale of current frame
    detected_faces = face_cascade.detectMultiScale(gray_scale)

    #Draw rectangle over current frame
    for (column, row, width, height) in detected_faces:
        cv2.rectangle(
            frame,
            (column, row),
            (column + width, row + height),
            (0, 255, 0),
            2
        )

        face_region= gray_scale[row : row+width, column : column+height]
        face_region=cv2.resize(face_region,(48,48))  
        img_arr = tf.keras.preprocessing.image.img_to_array(face_region)  
        img_arr = np.expand_dims(img_arr, axis = 0)  
        img_arr /= 255  
  
        predictions = model.predict(img_arr) 

        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index] 
        cv2.putText(frame, predicted_emotion, (int(row), int(column)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    return frame



def main():

    #Cascade of classifers that detect faces based on Haar features
    face_cascade = face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #Trained CNN model
    model = ks.models.load_model('CNN_lowest')

    #Capture video from webcam
    vidstream = cv2.VideoCapture('http://10.0.0.142:4747/video') #cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        #Get frame -> to gray scale -> detect faces
        _, frame = vidstream.read()
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = detect(gray_scale, frame, face_cascade, model)

        #Show result frame
        cv2.imshow('Detected', frame)

        #Wait for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            
    exit()        
            
    
        
        

if __name__ == "__main__":
    main()




