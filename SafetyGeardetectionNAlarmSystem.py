#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ultralytics import YOLO
import cv2
import math
import os


# In[5]:


from telegram import Bot
import requests

# Your Telegram Bot Token
def alert(l):
    token ='7091226766:AAFXsC0bNQiGBclfYduG1iQrRUh7YgDp4KM'
    chat_id='1960529010'
    # Initialize Telegram Bot
    message='Alert!!!!Wear {} immediately!!!!'.format(l)
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
    r=requests.get(url)
    print(r.json())



# In[6]:


# Load YOLO model with the specified weights file
model = YOLO("C:\\Users\\allen\\Downloads\\ppe.pt")

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # Use default camera (change index if needed)

# Define class names
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask',
              'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus',
              'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi',
              'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
reason=""
while True:
    # Read a frame from the webcam
    success, img = cap.read()
    if not success:
        break

    # Perform object detection with YOLO on the current frame
    results = model(img, stream=True)

    # Process the results of object detection
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract class label and confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Update color based on compliance
            if conf > 0.5:
                if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                    reason = currentClass.split('-')[1]  # Extract reason (e.g., "No Hardhat")
                    cv2.putText(img, f'{reason} ({conf})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                alert(reason)

    # Display the processed frame
    cv2.imshow("Live Detection", img)
     
   
    # Check for 'q' key press
    key = cv2.waitKey(1)
    if key == ord("q") or key == ord("Q"):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()


# In[ ]:




