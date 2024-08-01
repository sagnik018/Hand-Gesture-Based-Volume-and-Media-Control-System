import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np
import pyautogui
import time

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pycaw for volume control.
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Function to get volume range and initial volume
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Define the regions for gesture detection
volume_box_top_left = (70, 100)
volume_box_bottom_right = (220, 400)
gesture_box_top_left = (300, 100)
gesture_box_bottom_right = (570, 400)

# Function to count the number of fingers up
def count_fingers(lst):
    cnt = 0
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 5:
        cnt += 1

    return cnt

# Initialize variables for gesture detection
prev = -1
start_init = False
start_time = 0

# Capture video from the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Draw the gesture detection boxes
    cv2.rectangle(frame, volume_box_top_left, volume_box_bottom_right, (0, 0, 255), 3)
    cv2.rectangle(frame, gesture_box_top_left, gesture_box_bottom_right, (0, 255, 0), 3)
    
    # Process the frame and find hands
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get coordinates for thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            
            # Convert the normalized coordinates to pixel values
            h, w, _ = frame.shape
            thumb_tip = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_tip = (int(index_tip.x * w), int(index_tip.y * h))
            wrist = (int(wrist.x * w), int(wrist.y * h))
            middle_finger_mcp = (int(middle_finger_mcp.x * w), int(middle_finger_mcp.y * h))
            
            # Calculate the reference distance
            reference_distance = np.linalg.norm(np.array(wrist) - np.array(middle_finger_mcp))
            
            # Calculate the distance between the thumb and index finger
            distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
            
            # Calculate the normalized distance
            normalized_distance = distance / reference_distance
            
            # Check if the thumb tip and index tip are inside the volume box
            if (volume_box_top_left[0] < thumb_tip[0] < volume_box_bottom_right[0] and
                volume_box_top_left[1] < thumb_tip[1] < volume_box_bottom_right[1] and
                volume_box_top_left[0] < index_tip[0] < volume_box_bottom_right[0] and
                volume_box_top_left[1] < index_tip[1] < volume_box_bottom_right[1]):
                
                # Map the normalized distance to volume range
                volume_level = np.interp(normalized_distance, [0.2, 1.0], [min_vol, max_vol])
                volume.SetMasterVolumeLevel(volume_level, None)

                # Calculate volume percentage
                volume_percentage = np.interp(volume_level, [min_vol, max_vol], [0, 100])

                # Draw a circle at the tips and a line between them
                cv2.circle(frame, thumb_tip, 10, (0, 255, 0), -1)
                cv2.circle(frame, index_tip, 10, (0, 255, 0), -1)
                cv2.line(frame, thumb_tip, index_tip, (255, 0, 0), 4)
                
                # Display volume percentage
                cv2.putText(frame, f'Volume %: {int(volume_percentage)}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
                
                # Draw the volume bar
                bar_height = int(volume_percentage / 100 * 300)
                cv2.rectangle(frame, (50, 400 - bar_height), (70, 400), (0, 255, 0), -1)
                cv2.rectangle(frame, (50, 100), (70, 400), (0, 0, 255), 3)
            
            # Check if the thumb tip and index tip are inside the gesture box
            if (gesture_box_top_left[0] < thumb_tip[0] < gesture_box_bottom_right[0] and
                gesture_box_top_left[1] < thumb_tip[1] < gesture_box_bottom_right[1] and
                gesture_box_top_left[0] < index_tip[0] < gesture_box_bottom_right[0] and
                gesture_box_top_left[1] < index_tip[1] < gesture_box_bottom_right[1]):
                
                # Count fingers and perform actions
                cnt = count_fingers(hand_landmarks)
                if prev != cnt:
                    if not start_init:
                        start_time = time.time()
                        start_init = True
                    elif (time.time() - start_time) > 0.1:
                        action_text = ""
                        if cnt == 1:
                            pyautogui.press("left")
                            action_text = "Rewind"
                        elif cnt == 2:
                            pyautogui.press("right")
                            action_text = "Forward"
                        elif cnt == 3:
                            pyautogui.press("up")
                            action_text = "Volume Up"
                        elif cnt == 4:
                            pyautogui.press("down")
                            action_text = "Volume Down"
                        elif cnt == 5:
                            pyautogui.press("space")
                            action_text = "Play/Pause"
                        cv2.putText(frame, action_text, (360, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                        prev = cnt
                        start_init = False

    # Display the frame
    cv2.imshow('Gesture Control', frame)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()