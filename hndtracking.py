import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,          # Track up to 2 hands
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image color from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(img_rgb)

    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Optional: Print landmark positions
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Draw circle on the first landmark (wrist)
                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

    # Show the output
    cv2.imshow("Hand Tracking", img)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
