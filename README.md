# 👁️‍🗨️ Vision Projects Repository

Welcome to **Vision_Projects**, a collection of beginner-to-intermediate level Computer Vision projects developed using **OpenCV**, **MediaPipe**, and other powerful libraries. These projects demonstrate real-time applications such as face detection, hand tracking, gesture-based volume control, and attendance marking using face recognition.

Whether you're a curious beginner or an aspiring developer, these mini-projects will help you understand how powerful computer vision can be! 💻📷

---

## 🔍 Projects Included

### 1. 👤 Face Detection  
Detects human faces in real-time using Haar Cascades or MediaPipe.  
🛠️ **Tools**: OpenCV, HaarCascade  
✅ **Status**: Completed  
📸 **Screenshot**:  
![Face Detection](screenshots/face_detection.png)

---

### 2. 👁️ Face Mesh  
Maps 468 facial landmarks on the face, useful for AR filters, emotion detection, etc.  
🛠️ **Tools**: MediaPipe Face Mesh  
✅ **Status**: Completed  
📸 **Screenshot**:  
![Face Mesh](screenshots/face_mesh.png)

---

### 3. ✋ Hand Tracking  
Tracks hand landmarks in real-time and detects gestures.  
🛠️ **Tools**: MediaPipe Hands  
✅ **Status**: Completed  
📸 **Screenshot**:  
![Hand Tracking](screenshots/hand_tracking.png)

---

### 4. 🔊 Volume Gesture Control  
Control your system volume using finger gestures by measuring the distance between thumb and index finger.  
🛠️ **Tools**: MediaPipe + Pycaw + PyAutoGUI  
✅ **Status**: Completed  
📸 **Screenshot**:  
![Volume Gesture](screenshots/volume_gesture.png)

---

### 5. 📋 Attendance System (Face Recognition)  
Mark attendance by recognizing faces using webcam input.  
🧠 Face Encoding + CSV Logging  
🛠️ **Tools**: OpenCV, dlib, face_recognition, NumPy  
🚧 **Status**: _In Development..._  
👨‍🔧 Work in progress – stay tuned for updates!

---

## 📚 Libraries Used

This project uses the following Python libraries:

- 🔵 **OpenCV** – Image processing, object detection, and camera operations  
  ```bash
pip install opencv-python
pip install mediapipe
pip install numpy
pip install pycaw
pip install dlib
pip install pyautogui
pip install face_recognition


## 📁 Folder Structure

Vision_projects/
├── face_detection/
├── face_mesh/
├── hand_tracking/
├── volume_gesture/
├── attendance_project/     # Under development
└── screenshots/            # Screenshots used in README

🌟 Future Enhancements
Add GUI to attendance project

Add gesture-based game controls

Live face recognition with attendance timestamp

Add Pose estimation project

🙌 Contributing
Pull requests and feedback are welcome!
If you'd like to contribute, feel free to fork this repository and submit a PR.



