import math
import pyautogui
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

screen_width, screen_height = pyautogui.size()

def dis_curve(ax, ay, az, bx, by, bz, ox, oy, oz):
  dis1 = (ax - bx)**2 + (ay - by)**2 + (az - bz)**2
  dis2 = (ox - bx)**2 + (oy - by)**2 + (oz - bz)**2
  dis3 = (ax - ox)**2 + (ay - oy)**2 + (az - oz)**2
  cos = (dis1 + dis2 - dis3) / (2 * dis1**1/2 * dis2**1/2)
  return cos
def zjg(az, bz):
  return az > bz

def major1(ax, ay, bx, by, ox, oy):
  dis1 = (ax - bx) ** 2 + (ay - by) ** 2
  dis2 = (ox - bx) ** 2 + (oy - by) ** 2
  dis3 = (ax - ox) ** 2 + (ay - oy) ** 2
  cos = (dis1 + dis2 - dis3) / (2 * dis1 ** 1 / 2 * dis2 ** 1 / 2)
  return cos

def dis(ax, ay, bx, by):
  return (ax - bx) ** 2 + (ay - by) ** 2

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    ) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_world_landmarks:
        landmarks = hand_landmarks.landmark
        ax = landmarks[4].x
        ay = landmarks[4].y
        bx = landmarks[16].x 
        by = landmarks[16].y
        ox = landmarks[0].x
        oy = landmarks[0].y
        oz = landmarks[0].z
        # jz = [landmarks[i].z for i in [8, 12, 16, 20]]
        # rz = [landmarks[i].z for i in [5, 9, 13, 17]]
        # jr = [zjg(jz, rz) for i in range(4)]
        # print(jr)
        # print(sum(jr))
        jz = [results.multi_hand_world_landmarks[0].landmark[i].z for i in [7, 11, 15, 19]]
        rz = [results.multi_hand_world_landmarks[0].landmark[i].z for i in [5, 9, 13, 17]]
        jy = [results.multi_hand_landmarks[0].landmark[i].y for i in [8, 12, 16, 20]]
        ry = [results.multi_hand_landmarks[0].landmark[i].y for i in [5, 9, 13, 17]]
        l = len(jz)
        jr = [jz > rz for i in range(l)]
        print(sum(jr) == 0)
        print(sum([x - y for x, y in zip(jy, ry)])/4)
    # Flip the image horizontally for a selfie-view display.
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()