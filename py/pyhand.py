import os
import threading
import time
from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import pyautogui
import collections
from threading import Thread
import queue
from flask import Flask, request, jsonify
import json
import os

save_dir = 'D:/HandTrackingControl/'
shot_save_dir = 'D:/HandTrackingControl/shot/'
config_file = "config.json"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(shot_save_dir):
    os.mkdir(shot_save_dir)

app = Flask(__name__)
pyautogui.PAUSE = 0.01
screen_width, screen_height = pyautogui.size()
mp_hands = mp.solutions.hands

pyautogui.FAILSAFE = True
history_length = 5
x_history = collections.deque(maxlen=history_length)
y_history = collections.deque(maxlen=history_length)

screen_history = collections.deque(maxlen=history_length * 2)
drag_history = collections.deque(maxlen=history_length * 2)

HAND_TYPES = ['', 'Right', 'Left']

MAIN_HAND = 1
x_fix = 0.2
y_fix = 0.65
sensitivity = 4
click_sensitivity = 0.06
scroll_value = 200
action1 = ['']
action2 = ['']
action3 = ['']

class VideoStream:
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.stopped = True
        self.frame_queue = queue.Queue(maxsize=1)
        self.thread = None

    def start(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.stopped = False

        if self.thread is None or not self.thread.is_alive():
            self.thread = Thread(target=self.update, args=())
            self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def read(self):
        return self.frame_queue.get()

    def stop(self):
        self.stopped = True
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.thread is not None:
            try:
                self.thread.join()
            except:
                print("cant join thread")
            self.thread = None
        while not self.frame_queue.empty():
            self.frame_queue.get()

def load_config():
    if os.path.exists(save_dir + config_file):
        with open(save_dir + config_file, "r") as f:
            return json.load(f)
    return {
        "x_fix": 0.2,
        "y_fix": 0.65,
        "sensitivity": 4,
        "click_sensitivity": 0.04,
        "scroll_value": 200,
        "hand": 1,
        "action1": "",
        "action2": "",
        "action3": ""
    }

def save_config(config):
    with open(save_dir + config_file, "w") as f:
        json.dump(config, f, indent=4)

def is_hand_label_match(label, main_hand):
    return label == HAND_TYPES[-main_hand]

def should_click(y8, pre_y, y5):
    return y8 - pre_y > click_sensitivity and y8 - y5 > -0.02

def move_cursor(current_x, current_y):
    x_history.append(current_x)
    y_history.append(current_y)
    smooth_x = sum(x_history) / len(x_history)
    smooth_y = sum(y_history) / len(y_history)
    moveto_x = int((1 - (smooth_x - x_fix) * 4) * screen_width)
    moveto_y = int((smooth_y - y_fix) * 4 * screen_height)

    moveto_x = max(1, min(moveto_x, screen_width - 2))
    moveto_y = max(1, min(moveto_y, screen_height - 2))

    pyautogui.moveTo(moveto_x, moveto_y)

def drag_cursor(current_x, current_y):
    x_history.append(current_x)
    y_history.append(current_y)
    smooth_x = sum(x_history) / len(x_history)
    smooth_y = sum(y_history) / len(y_history)
    moveto_x = int((1 - (smooth_x - x_fix) * 4) * screen_width)
    moveto_y = int((smooth_y - 0.65) * 4 * screen_height)

    moveto_x = max(1, min(moveto_x, screen_width - 2))
    moveto_y = max(1, min(moveto_y, screen_height - 2))
    cx, cy = pyautogui.position()
    pyautogui.mouseDown()
    pyautogui.move((moveto_x - cx) / 2, (moveto_y - cy) / 2)

def is_disable_condition(x5, x17, is_right_hand):
    if is_right_hand:
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

def is_drag(landmarks, jz2, rz, drag_history):
    ax = landmarks[4].x
    ay = landmarks[4].y
    bx = landmarks[16].x
    by = landmarks[16].y
    jy = [landmarks[i].y for i in [8, 12, 16, 20]]
    ry = [landmarks[i].y for i in [5, 9, 13, 17]]
    drag = dis(ax, ay, bx, by) < 0.003 and 0 < sum([x - y for x, y in zip(jy, ry)]) / 4 <= 0.12 and sum(
        [jz2 > rz for i in range(4)]) == 0
    drag_history.append(drag)
    return drag

def dis(ax, ay, bx, by):
  return (ax - bx) ** 2 + (ay - by) ** 2

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
    count = 0
    pre_y12r = -1
    pre_y8r = -1
    pre_x4r = -1
    is_shot = False
    last_time = time.time()
    hand_need_set_state = True
    shot_time = time.time()
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
                            pyautogui.mouseUp()
                            current_hand = -1
                            try_jg_l = 0
                            try_jg_r = 0
                        else:
                            try_jg_r += 1
                            continue
                    jz = [results.multi_hand_world_landmarks[idx].landmark[i].z for i in [8, 12, 16, 20]]
                    jz2 = [results.multi_hand_world_landmarks[idx].landmark[i].z for i in [7, 11, 15, 19]]
                    rz = [results.multi_hand_world_landmarks[idx].landmark[i].z for i in [5, 9, 13, 17]]
                    y8 = landmarks[8].y
                    y5 = landmarks[5].y
                    x5 = landmarks[5].x
                    x17 = landmarks[17].x
                    y12 = landmarks[12].y
                    y9 = landmarks[9].y
                    x4 = landmarks[4].x
                    x9 = landmarks[9].x
                    if is_drag(landmarks, jz2, rz, screen_history):
                        if is_shot or time.time() - shot_time < 1:
                            continue
                        else:
                            is_shot = True
                            shot_time = time.time()
                            pyautogui.screenshot().save(shot_save_dir +
                                                        time.ctime(time.time()
                                                                   ).replace(':', '_').replace(' ', '_') + '.png')
                            continue
                    if(sum(screen_history) == 0):
                        is_shot = False
                    if hand_need_set_state:
                        l = len(jz)
                        jr = [z_sroll(jz, rz) for i in range(l)]
                        if sum(jr) == l or sum(jr) == 0:
                            hand_state = (sum(jr) == l)
                            hand_need_set_state = False
                        continue
                    elif do_scroll(jz, rz, hand_state):
                        if reset:
                            pyautogui.scroll(scroll_value * MAIN_HAND if hand_state else -scroll_value * MAIN_HAND)
                            reset = False
                            last_time = time.time()
                        elif time.time() - last_time >= 1 and not reset:
                            hand_state = not hand_state
                            reset = True
                        continue
                    else:
                        reset = True
                    if should_click(y8, pre_y8r, y5):
                        pyautogui.hotkey(*action2)
                    elif should_click(y12, pre_y12r, y9):
                        pyautogui.hotkey(*action3)
                    elif is_disable_condition(x4, (x5 + landmarks[7].x)/2, label=='Left'
                                            ) and abs(x4 - pre_x4r) > abs(x5 - x9):
                        pyautogui.hotkey(*action1)
                    pre_y12r = y12
                    pre_y8r = y8
                    pre_x4r = x4
                    continue

                if disable:
                    x5 = hand_landmarks.landmark[5].x
                    x17 = hand_landmarks.landmark[17].x
                    if is_disable_condition(x5, x17, label=='Left'):
                        act = True
                        continue
                    if not is_disable_condition(x5, x17, label=='Left') and act:
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
                    jz2 = [results.multi_hand_world_landmarks[idx].landmark[i].z for i in [7, 11, 15, 19]]
                    rz = [results.multi_hand_world_landmarks[idx].landmark[i].z for i in [5, 9, 13, 17]]
                    if is_disable_condition(x5, x17, label=='Left'):
                        disable = True
                        continue
                    if is_drag(landmarks, jz2, rz, drag_history) or sum(drag_history) != 0:
                        drag_cursor(total_x / 5, total_y / 5)
                        continue
                    else:
                        pyautogui.mouseUp()
                    if is_disable_condition(x4, (x5 + landmarks[7].x)/2, label=='Left'
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

hand_tracking_thread = None
stream = VideoStream(0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    global hand_tracking_thread
    global y_fix
    global x_fix
    global sensitivity
    global click_sensitivity
    global scroll_value
    global MAIN_HAND
    global action1
    global action2
    global action3
    data = request.json
    x_fix = data.get("x_fix")
    y_fix = data.get("y_fix")
    sensitivity = data.get("sensitivity")
    click_sensitivity = 0.1 - data.get("click_sensitivity")
    scroll_value = data.get("scroll_value")
    action1 = data.get("action1").split('+')
    action2 = data.get("action2").split('+')
    action3 = data.get("action3").split('+')
    save_config({
        "x_fix": x_fix,
        "y_fix": y_fix,
        "sensitivity": sensitivity,
        "click_sensitivity": click_sensitivity,
        "scroll_value": scroll_value,
        "action1": data.get("action1"),
        "action2": data.get("action2"),
        "action3": data.get("action3"),
        "hand": MAIN_HAND
    })
    if stream.stopped:
        stream.start()
        hand_tracking_thread = Thread(target=hand_tracking)
        hand_tracking_thread.start()
        return jsonify({"status": "已启动"})
    return jsonify({"status": "正在运行"})

@app.route("/stop", methods=["POST"])
def stop():
    global stream
    stream.stop()
    return jsonify({"status": "已停止"})

@app.route("/set_hand", methods=["POST"])
def set_hand():
    global MAIN_HAND
    data = request.json
    MAIN_HAND = data.get("hand", 1)
    return jsonify({"status": "hand set", "hand": MAIN_HAND})

@app.route("/get_config", methods=["GET"])
def get_config():
    config = load_config()
    return jsonify(config)

if __name__ == "__main__":
    app.run(debug=True)