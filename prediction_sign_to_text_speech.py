import math
import cv2
import numpy as np
import traceback
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model

# -------------------- LOAD MODEL --------------------
model = load_model("cnn8grps_rad1_model.h5")

# -------------------- TEXT TO SPEECH --------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)

# -------------------- HAND DETECTORS --------------------
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# -------------------- WHITE CANVAS --------------------
white = np.ones((400, 400, 3), dtype=np.uint8) * 255

offset = 30
last_spoken = ""
text_output = ""

# -------------------- DISTANCE FUNCTION --------------------
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# -------------------- MAIN LOOP --------------------
while True:
    try:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            crop = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            white[:] = 255

            hand2 = hd2.findHands(crop, draw=False, flipType=True)

            if hand2:
                lm = hand2[0]['lmList']

                osx = (400 - w) // 2
                osy = (400 - h) // 2

                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (5,6),(6,7),(7,8),
                    (9,10),(10,11),(11,12),
                    (13,14),(14,15),(15,16),
                    (17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17),(0,5),(0,17)
                ]

                for a,b in connections:
                    cv2.line(
                        white,
                        (lm[a][0]+osx, lm[a][1]+osy),
                        (lm[b][0]+osx, lm[b][1]+osy),
                        (0,255,0), 2
                    )

                for i in range(21):
                    cv2.circle(
                        white,
                        (lm[i][0]+osx, lm[i][1]+osy),
                        3, (0,0,255), -1
                    )

                cv2.imshow("Skeleton", white)

                img = white.reshape(1,400,400,3)
                prob = model.predict(img, verbose=0)[0]
                ch = np.argmax(prob)

                # -------- GROUP TO LETTER --------
                mapping = {
                    0:'A', 1:'B', 2:'C', 3:'G',
                    4:'L', 5:'P', 6:'X', 7:'Y'
                }
                letter = mapping[ch]

                # -------- SPACE --------
                if letter in ['E','S','X','Y','B']:
                    if lm[6][1] > lm[8][1] and lm[18][1] > lm[20][1]:
                        letter = ' '

                # -------- NEXT --------
                if letter in ['A','B','Y']:
                    if lm[4][0] < lm[5][0]:
                        letter = 'NEXT'

                # -------- BACKSPACE --------
                if letter in ['NEXT','B','C','G','P']:
                    if lm[0][0] > lm[8][0]:
                        letter = 'BACK'

                # -------- TEXT LOGIC --------
                if letter == 'NEXT':
                    pass

                elif letter == 'BACK':
                    text_output = text_output[:-1]

                else:
                    if letter != last_spoken:
                        text_output += letter
                        engine.say(letter if letter != ' ' else "space")
                        engine.runAndWait()
                        last_spoken = letter

                cv2.putText(
                    frame,
                    f"Text: {text_output}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,0,255), 3
                )

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception:
        print(traceback.format_exc())

cap.release()
cv2.destroyAllWindows()
