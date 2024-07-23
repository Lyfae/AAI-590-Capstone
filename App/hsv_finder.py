import cv2
import numpy as np

def nothing(x):
    pass

# Open the webcam
cap = cv2.VideoCapture(0)

# Create a window
cv2.namedWindow('Trackbars')

# Create trackbars for color change
cv2.createTrackbar('H Lower', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('S Lower', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('V Lower', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('H Upper', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('S Upper', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('V Upper', 'Trackbars', 0, 255, nothing)

# Set initial values for the upper HSV trackbars
cv2.setTrackbarPos('H Upper', 'Trackbars', 179)
cv2.setTrackbarPos('S Upper', 'Trackbars', 255)
cv2.setTrackbarPos('V Upper', 'Trackbars', 255)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if the frame is successfully captured
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of the trackbars
    h_lower = cv2.getTrackbarPos('H Lower', 'Trackbars')
    s_lower = cv2.getTrackbarPos('S Lower', 'Trackbars')
    v_lower = cv2.getTrackbarPos('V Lower', 'Trackbars')
    h_upper = cv2.getTrackbarPos('H Upper', 'Trackbars')
    s_upper = cv2.getTrackbarPos('S Upper', 'Trackbars')
    v_upper = cv2.getTrackbarPos('V Upper', 'Trackbars')

    # Define the HSV range based on the trackbar positions
    lower_hsv = np.array([h_lower, s_lower, v_lower])
    upper_hsv = np.array([h_upper, s_upper, v_upper])

    # Create a binary mask where white represents the colors within the range
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show the original frame and the result
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the c
