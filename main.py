import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Variables to manage click timing
click_threshold = 20
scroll_threshold = 20
last_click_time = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            index_x, index_y = None, None
            thumb_x, thumb_y = None, None

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                if id == 8:  # Index finger tip
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y

                if id == 4:  # Thumb tip
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y

            # Perform scrolling based on vertical movement
            if index_y is not None and thumb_y is not None:
                if thumb_y < index_y - scroll_threshold:
                    # Scroll up
                    pyautogui.scroll(10)
                elif thumb_y > index_y + scroll_threshold:
                    # Scroll down
                    pyautogui.scroll(-10)

            # Perform click if thumb is close to index finger vertically
            if thumb_x is not None and thumb_y is not None and index_x is not None and index_y is not None:
                if abs(index_y - thumb_y) < click_threshold:
                    current_time = cv2.getTickCount() / cv2.getTickFrequency()
                    if current_time - last_click_time > 1:  # Adding 1 second delay between clicks
                        pyautogui.click()
                        last_click_time = current_time

                # Move cursor smoothly
                else:
                    pyautogui.moveTo(index_x, index_y)

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
