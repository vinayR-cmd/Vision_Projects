import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime 

path="attendenceproject/attendeemages"
images=[]
classnames=[]
mylist=os.listdir(path)
print(mylist)
for cl in mylist:
    curimg=cv2.imread(f'{path}/{cl}') 
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)


def findencodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

# def markattendence(name):
#     with open("attendence.csv","r+") as f:
#         mydatalist=f.readlines()
#         namelist=[]
#         for line in mydatalist:
#             entry=line.split(',')
#             namelist.append(entry[0])
#         if name not in namelist:
#             now=datetime.now()
#             dtstring=now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtstring}')

def markattendence(name):
    # Create the file if it doesn't exist
    if not os.path.exists("attendence.csv"):
        with open("attendence.csv", "w") as f:
            f.write("Name,Time\n")

    with open("attendenceproject/attendence.csv", "r+") as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.strip().split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')



# markattendence(name)

encodelistknown=findencodings(images)
print("Encoding complete")

cap=cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facescurrframe = face_recognition.face_locations(imgs)
    encodescurrframe = face_recognition.face_encodings(imgs, facescurrframe)

    for encodeface, faceloc in zip(encodescurrframe, facescurrframe):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)
        facedis = face_recognition.face_distance(encodelistknown, encodeface)
        matchindex = np.argmin(facedis)

        if matches[matchindex]:
            name = classnames[matchindex].upper()
            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markattendence(name)

    cv2.imshow("webcam", img)

    # 👇 Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release resources
cap.release()
cv2.destroyAllWindows()
