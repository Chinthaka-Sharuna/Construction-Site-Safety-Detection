import cv2
from ultralytics import YOLO
import cvzone
import math



cap = cv2.VideoCapture("./data/ppe-3.mp4")
# cap=cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

model=YOLO('best.pt')
model.to('cuda')


while True:
    success, frame = cap.read()

    if not success:
        break

    results=model(frame,stream=True)

    for result in results:
        for box in result.boxes:
            x1,y1,x2,y2=box.xyxy[0]
            bbox=[int(x1),int(y1),int(x2-x1),int(y2-y1)]


            cvzone.cornerRect(frame,bbox)
            cvzone.putTextRect(frame,f"{model.names[int(box.cls[0])]} {math.ceil((box.conf[0]*100))/100}",(max(0,int(x1)),int(y1)),scale=1,thickness=1)


    cv2.imshow("frame",frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()




