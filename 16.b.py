import cv2

# Load pre-trained classifier for full body detection
full_cascade = cv2.CascadeClassifier('E:\Computer Vision stuff\OpenCV\haarcascades\Haarcascades\haarcascade_smile.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for full body detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect full bodies in the frame
    full_bodies = full_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # For each detected full body, draw a red rectangle around it
    for (x, y, w, h) in full_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the frame with full body detections
    cv2.imshow('Full Body Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
