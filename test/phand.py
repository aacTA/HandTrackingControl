import threading
import time
from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import pyautogui
import collections
from threading import Thread
import queue

app = Flask(__name__)
pyautogui.PAUSE = 0.01
screen_width, screen_height = pyautogui.size()
mp_hands = mp.solutions.hands

pyautogui.FAILSAFE = True
history_length = 5
x_history = collections.deque(maxlen=history_length)
y_history = collections.deque(maxlen=history_length)

HAND_TYPES = ['', 'Right', 'Left']
MAIN_HAND = 1


class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stopped = False
        self.frame_queue = queue.Queue(maxsize=1)

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def read(self):
        return self.frame_queue.get()

    def stop(self):
        self.stopped = True
        self.cap.release()


def is_hand_label_match(label, main_hand):
    return label == HAND_TYPES[-main_hand]

def should_click(y8, pre_y, y5):
    return y8 - pre_y > 0.06 and y8 - y5 > -0.02

def move_cursor(current_x, current_y):
    x_history.append(current_x)
    y_history.append(current_y)
    smooth_x = sum(x_history) / len(x_history)
    smooth_y = sum(y_history) / len(y_history)
    moveto_x = int((1 - (smooth_x - 0.2) * 4) * screen_width)
    moveto_y = int((smooth_y - 0.75) * 4 * screen_height)

    moveto_x = max(1, min(moveto_x, screen_width - 2))
    moveto_y = max(1, min(moveto_y, screen_height - 2))

    pyautogui.moveTo(moveto_x, moveto_y)

def is_disable_condition(x5, x17, is_main_hand):
    if is_main_hand:
        return x5 < x17
    else:
        return x5 > x17

def z_sroll(az, bz):
  return az > bz

def do_scroll(jz, rz, state):
    for i in range(len(jz)):
        if z_sroll(jz[i], rz[i]) == state:
            return False
    return True

def angel():
    degree = 45
    return degree

def is_drag():

    return True

def hand_tracking():
    hand_state = True
    stream.start()
    pre_y8 = -1
    pre_y12 = -1
    pre_x4 = -1
    disable = False
    act = False
    reset = True
    current_hand = 1
    try_jg_l = 0
    try_jg_r = 0
    last_time = time.time()
    hand_need_set_state = True
    with (mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands):
        while not stream.stopped:
            if stream.frame_queue.empty():
                continue

            image = stream.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            if not results.multi_hand_landmarks:
                continue
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                is_main_hand = is_hand_label_match(label, MAIN_HAND)
                landmarks = hand_landmarks.landmark
                if not is_main_hand:

                    if len(results.multi_hand_landmarks) == 1:
                        if current_hand == -1 or try_jg_r >= 10:
                            act = False
                            disable = True
                            current_hand = -1
                            try_jg_l = 0
                            try_jg_r = 0
                        else:
                            try_jg_r += 1
                            continue

                    jz = [results.multi_hand_world_landmarks[idx].landmark[i].z for i in [8, 12, 16, 20]]
                    rz = [results.multi_hand_world_landmarks[idx].landmark[i].z for i in [5, 9, 13, 17]]

                    if hand_need_set_state:
                        l = len(jz)
                        jr = [z_sroll(jz, rz) for i in range(l)]
                        if sum(jr) == l or sum(jr) == 0:
                            hand_state = (sum(jr) == l)
                            print('set_newstate')
                            hand_need_set_state = False
                    elif do_scroll(jz, rz, hand_state):
                        if reset:
                            pyautogui.scroll(200 if hand_state else -200)
                            reset = False
                            last_time = time.time()
                        elif time.time() - last_time >= 1 and not reset:
                            print(hand_state)
                            hand_state = not hand_state
                            print(hand_state)
                            reset = True
                    else:
                        reset = True
                    continue
                if len(results.multi_hand_landmarks) == 1:
                    if current_hand == 1 or try_jg_l >= 10:
                        print(results.multi_handedness[idx].classification[0].score)
                        hand_need_set_state = True
                        reset = True
                        current_hand = 1
                        try_jg_l = 0
                        try_jg_r = 0
                    else:
                        try_jg_l += 1
                        continue

                if disable:
                    x5 = hand_landmarks.landmark[5].x
                    x17 = hand_landmarks.landmark[17].x
                    if is_disable_condition(x5, x17, is_main_hand):
                        act = True
                        continue
                    if not is_disable_condition(x5, x17, is_main_hand) and act:
                        act = False
                        disable = False
                        continue
                else:
                    total_x = sum(landmarks[i].x for i in [0, 5, 9, 13, 17])
                    total_y = sum(landmarks[i].y for i in [0, 5, 9, 13, 17])
                    y8 = landmarks[8].y
                    y5 = landmarks[5].y
                    x5 = landmarks[5].x
                    x17 = landmarks[17].x
                    y12 = landmarks[12].y
                    y9 = landmarks[9].y
                    x4 = landmarks[4].x
                    x9 = landmarks[9].x
                    if is_disable_condition(x5, x17, is_main_hand):
                        disable = True
                        continue
                    if is_disable_condition(x4, (total_x-landmarks[0].x - landmarks[17].x)/3, is_main_hand
                                            ) and abs(x4 - pre_x4) > abs(x5 - x9):
                        pyautogui.doubleClick()
                    elif should_click(y8, pre_y8, y5):
                        pyautogui.click()
                    elif should_click(y12, pre_y12, y9):
                        pyautogui.rightClick()
                    else:
                        move_cursor(total_x / 5, total_y / 5)
                    pre_y12 = y12
                    pre_y8 = y8
                    pre_x4 = x4
            if cv2.waitKey(1) & 0xFF == 27:
                stream.stop()
                break

    cv2.destroyAllWindows()

hand_tracking_thread = Thread(target=hand_tracking)
stream = VideoStream(0)
# Web 路由
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global hand_tracking_thread
    if hand_tracking_thread is None or not hand_tracking_thread.is_alive():
        hand_tracking_thread = Thread(target=hand_tracking)
        hand_tracking_thread.start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})

@app.route("/stop", methods=["POST"])
def stop():
    global stream
    if stream is not None:
        stream.stop()
    return jsonify({"status": "stopped"})

@app.route("/set_hand", methods=["POST"])
def set_hand():
    global MAIN_HAND
    data = request.json
    MAIN_HAND = data.get("hand", 1)  # 1: 右手, 2: 左手
    return jsonify({"status": "hand set", "hand": MAIN_HAND})

if __name__ == "__main__":
    app.run(debug=True)