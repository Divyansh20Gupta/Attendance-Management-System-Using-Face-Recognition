import requests
import cv2
import numpy as np

url = "http://192.168.0.6:8080/shot.jpg"

while True:
    try:
        cam = requests.get(url)
        imgNp = np.array(bytearray(cam.content), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)

        cv2.imshow("IP Camera", img)

    except Exception as e:
        print("Error:", e)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

