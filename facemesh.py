import cv2
import mediapipe as mp
import time

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open webcam
cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    img = cv2.resize(img, (800, 800))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw mesh tesselation
            mp_draw.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

            # Optional: show first 5 landmarks
            for id, lm in enumerate(face_landmarks.landmark[:5]):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 3, (0, 255, 255), cv2.FILLED)
                cv2.putText(img, f'{id}', (cx+5, cy-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-5)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # Show image
    cv2.imshow("Face Mesh - Webcam", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()