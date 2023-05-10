import cv2

# for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# default webcam
stream = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    (grabbed, frame) = stream.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Try to detect faces in the webcam
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # for each faces found
    for (x, y, w, h) in faces:        
        # Draw a rectangle around the face
        color = (0, 255, 255) # in BGR
        stroke = 5    
        cv2.rectangle(frame, (x, y), (x + w, y + h), 
            color, stroke)

    # Show the frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF    
    if key == ord("q"):    # Press q to break out of the loop
        break

# Cleanup
stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)