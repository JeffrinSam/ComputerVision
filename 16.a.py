import cv2

# Load pre-trained classifiers for face, eye, and full body detection
face_cascade = cv2.CascadeClassifier('E:\Computer Vision stuff\OpenCV\haarcascades\Haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('E:\Computer Vision stuff\OpenCV\haarcascades\Haarcascades\haarcascade_eye.xml')
full_cascade = cv2.CascadeClassifier('E:\Computer Vision stuff\OpenCV\haarcascades\Haarcascades\haarcascade_fullbody.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face, eye, and full body detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect eyes within the region of interest (ROI)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    # Detect full bodies in the frame
    full_bodies = full_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in full_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the frame with face, eye, and full body detections
    cv2.imshow('Face, Eye, and Full Body Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()









