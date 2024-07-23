import cv2 as cv
import numpy as np
import time

# Open camera for fvideo capture
cap = cv.VideoCapture(0) 

# Time Counter
countdown_timer = time.time()
COUNTDOWN_DURATION = 5

# Pauls HSV Bounds
HSV_LOW = (0,0,193)
HSV_HIGH = (34,255,255)

# Crop Boundries
X,Y,W,H = 100,100,400,400

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

    # Crop the frame
    frame = frame[Y:Y+H,X:X+W]

    # Change color image to hsv
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    # Creating a mask based on the HSV threshold
    mask = cv.inRange(hsv,HSV_LOW, HSV_HIGH)

    # Find contours in the frame
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Filter out the smaller contours
    CONTOUR_MIN = 8000
    contours_filtered = [contour for contour in contours if cv.contourArea(contour) > CONTOUR_MIN]

    # Draw contours on the frame
    draw_contours = cv.drawContours(frame, contours_filtered, -1, (0,255,0), 2)

    # Create a new blank mask for contour areas
    contour_mask = np.zeros_like(mask) 

    # Draw filled contours on the new mask
    contour_mask = cv.drawContours(contour_mask, contours_filtered, -1, 255, cv.FILLED)

    # Get the current countdown time
    elapsed_time = time.time() - countdown_timer
    countdown_time = max(0, COUNTDOWN_DURATION - int(elapsed_time))

    # Display countdown on the frame
    cv.putText(frame, f"Countdown: {countdown_time}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Check if countdown has finished
    if countdown_time == 0:
        # Capture the frame
        captured_mask = contour_mask.copy()
        cv.imshow("Captured Frame", captured_mask)

        # Restart countdown
        countdown_timer = time.time()
    
    # Display the frame
    cv.imshow("Frame",frame)
    cv.imshow("Mask",contour_mask)

    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv.destroyAllWindows() 



