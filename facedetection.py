import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Face Detection
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

# Start webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (640, 480))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # Draw detection on the image
            mpDraw.draw_detection(img, detection)

            # Calculate and draw bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            cv2.rectangle(img, bbox, (255, 0, 0), 3)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 2)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-5)  # Added epsilon to prevent divide by 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Webcam Face Detection", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()