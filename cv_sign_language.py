import cv2 as cv
import numpy as np

# Open camera for fvideo capture
cap = cv.VideoCapture(0) 

# Check to see if camera can open
if not cap.isOpened():
    print("Error Opening Camera")
    exit(0)

while True:
    # Grab the frames
    ret, frame = cap.read()

    # Check to see if the frames are returned
    if not ret:
        print("Failed to grab the frames - aborting")
        break

    # Grab grayscale image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Binary threshold
    _, threshhold = cv.threshold(gray,130,255, cv.THRESH_BINARY)

    # Find contours in the frame
    contours, _ = cv.findContours(threshhold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours on the frame
    draw_contours = cv.drawContours(frame, contours, -1, (0,255,0),2)
    
    # Display the frame
    cv.imshow("Frame",frame)

    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv.destroyAllWindows() 



