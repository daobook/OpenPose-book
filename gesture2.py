import cv2
import time
from pathlib import Path
from collections import Counter
import numpy as np
from joblib import load
from mediapipe.python.solutions import drawing_utils, hands
import pyautogui

pyautogui.FAILSAFE= True

print('GESTURE RECOGNITION')
print("Press 'q' to quit")
print("Press 'd' for debug")

debug = False
t = 0
txt_offset = (-60, 30)

# load the models
gesture_model = load('./model/gesture_model.pkl')

hand_model = hands.Hands(static_image_mode=True,
                         min_detection_confidence=0.7,
                         min_tracking_confidence=0.7, max_num_hands=4)

# initialize video capture
vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
resolution = (vc.get(3), vc.get(4))
gesture_list = []
gesture_id = -1
n = 0
save_dir = '../screenshots'
_save_dir = Path(save_dir)
if not _save_dir.exists():
    _save_dir.mkdir(exist_ok=True)

while vc.isOpened():
    ret, frame = vc.read()
    # get hands model prediction
    results = hand_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            if debug:
                # draw hand skeleton
                drawing_utils.draw_landmarks(frame, handLandmarks,
                                             hands.HAND_CONNECTIONS)

            # get predicted points
            x, y = [], []
            for lm in handLandmarks.landmark:
                x.append(lm.x)
                y.append(lm.y)

            txt_pos = np.add(np.multiply(resolution, (x[0], y[0])), txt_offset)

            # normalize points
            points = np.asarray([x, y])
            min = points.min(axis=1, keepdims=True)
            max = points.max(axis=1, keepdims=True)
            normalized = np.stack((points-min)/(max-min), axis=1).flatten()

            # get prediction and confidence
            pred = gesture_model.predict_proba([normalized])
            gesture = pred.argmax(axis=1)[0]
            confidence = pred.max()
            gesture_id += 1
            # add text
            cv2.putText(frame, "'{}',{:.1%}".format(gesture, confidence),
                        txt_pos.astype(int), cv2.FONT_HERSHEY_DUPLEX,  1,
                        (0, 255, 255), 1, cv2.LINE_AA)
            gesture_list.append(int(gesture))
            counters = Counter(gesture_list)

            if len(gesture_list) < 18:
                print(gesture_list)  
                time.sleep(1.5)
                if len(gesture_list) > 16:
                    if 1 in gesture_list[-16:-12] and 4 in gesture_list[-12:-8] and 5 in gesture_list[-8:-4] and 0 in gesture_list[-4:]: # 1450
                        pyautogui.alert('请问医院在哪里？')
                    elif 1 in gesture_list[-16:-12] and 4 in gesture_list[-12:-8] and 5 in gesture_list[-8:-4] and 4 in gesture_list[-4:]: # 1454
                        pyautogui.alert('请问最近的超市在哪？')
                    elif 1 in gesture_list[-16:-12] and 4 in gesture_list[-12:-8] and 5 in gesture_list[-8:-4] and 1 in gesture_list[-4:]: # 1451
                        # pyautogui.confirm('顶级黑客事件')
                        pyautogui.alert('请问警察局在哪？')
                    elif 1 in gesture_list[-16:-12] and 4 in gesture_list[-12:-8] and 5 in gesture_list[-8:-4] and 2 in gesture_list[-4:]: # 1452
                        # pyautogui.confirm('顶级黑客事件')
                        pyautogui.alert('请问附近有没有餐厅？')
                    elif 1 in gesture_list[-16:-12] and 4 in gesture_list[-12:-8] and 5 in gesture_list[-8:-4] and 3 in gesture_list[-4:]: # 1453
                        # pyautogui.confirm('顶级黑客事件')
                        pyautogui.alert('请问附近的加油站在哪里？')
                    elif 1 in gesture_list[-16:-12] and 4 in gesture_list[-12:-8] and 5 in gesture_list[-8:-4] and 5 in gesture_list[-4:]: # 1455
                        pyautogui.alert('请问最近的火车站在哪？')
                    elif 1 in gesture_list[-16:-12] and 4 in gesture_list[-12:-8] and 5 in gesture_list[-8:-4] and 6 in gesture_list[-4:]: # 1456
                        # pyautogui.confirm('顶级黑客事件')
                        pyautogui.alert('请问最近的公交车站在哪？')
                    elif 1 in gesture_list[-16:-12] and 4 in gesture_list[-12:-8] and 5 in gesture_list[-8:-4] and 7 in gesture_list[-4:]: # 1457
                        # pyautogui.confirm('顶级黑客事件')
                        pyautogui.alert('请问市中心往那个方向走？')
                    elif 1 in gesture_list[-16:-12] and 4 in gesture_list[-12:-8] and 5 in gesture_list[-8:-4] and 8 in gesture_list[-4:]: # 1458
                        # pyautogui.confirm('顶级黑客事件')
                        pyautogui.alert('请问附近的银行怎么走？')
                    elif 1 in gesture_list[-16:-12] and 4 in gesture_list[-12:-8] and 5 in gesture_list[-8:-4] and 9 in gesture_list[-4:]: # 1459
                        # pyautogui.confirm('顶级黑客事件')
                        pyautogui.alert('请问最近的药店怎么走？')

                
            else:
                gesture_list = []

    fps = 1/(time.time()-t)
    t = time.time()

    # debug text
    if debug:
        cv2.putText(frame,
                    "{}x{}; {}fps".format(*resolution, int(fps)),
                    (0, 15),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    # get keyinput
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        vc.release()
    if key == ord('d'):
        debug = not debug

cv2.destroyAllWindows()
