import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils

# Initialize webcam
webcam = cv2.VideoCapture(0)

x1 = y1 = x2 = y2 = 0
prev_time = 0
cooldown = 0.5  # seconds

while True:
    _, img = webcam.read()
    img = cv2.flip(img, 1)  # Mirror image
    frame_height, frame_width, _ = img.shape

    # Convert image to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    output = hands.process(rgb_img)
    hand_landmarks = output.multi_hand_landmarks

    if hand_landmarks:
        for hand in hand_landmarks:
            drawing_utils.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index fingertip
                    cv2.circle(img, (x, y), 8, (0, 255, 255), -1)
                    x1, y1 = x, y

                if id == 4:  # Thumb tip
                    cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
                    x2, y2 = x, y

            # Draw line between thumb and index
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Calculate distance
            distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

            current_time = time.time()
            if current_time - prev_time > cooldown:
                if distance > 50:
                    pyautogui.press("volumeup")
                else:
                    pyautogui.press("volumedown")
                prev_time = current_time

    # Display the result
    cv2.imshow("Volume Control", img)

    # Break loop on 'p' key press
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

webcam.release()
cv2.destroyAllWindows()