import cv2
import mediapipe as mp
import math
import time
from pythonosc.udp_client import SimpleUDPClient

# Configura OSC
ip = "127.0.0.1"       # localhost
porta = 8000           # porta su cui TouchDesigner ascolta
client = SimpleUDPClient(ip, porta)

# Inizializza MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Avvia la webcam
cap = cv2.VideoCapture(0)

prev_state = None
last_send_time = time.time()

def distanza(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            h, w, _ = img.shape
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if len(lm_list) >= 9:
                thumb_tip = lm_list[4]
                index_tip = lm_list[8]

                # Disegno
                cv2.circle(img, thumb_tip, 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, index_tip, 10, (0, 255, 0), cv2.FILLED)
                cv2.line(img, thumb_tip, index_tip, (0, 255, 255), 3)

                # Calcolo distanza
                dist = int(distanza(thumb_tip, index_tip))
                cv2.putText(img, f'Distanza: {dist}px', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                # Stato aperto o chiuso
                threshold = 50
                current_time = time.time()
                if current_time - last_send_time > 0.5:
                    if dist > threshold and prev_state != "open":
                        prev_state = "open"
                        client.send_message("/mano/stato", 1)  # 1 = aperto
                        client.send_message("/mano/distanza", dist)
                        last_send_time = current_time
                    elif dist <= threshold and prev_state != "closed":
                        prev_state = "closed"
                        client.send_message("/mano/stato", 0)  # 0 = chiuso
                        client.send_message("/mano/distanza", dist)
                        last_send_time = current_time

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostra il video
    cv2.imshow("Tracking mano - OSC", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC per uscire
        break

cap.release()
cv2.destroyAllWindows()
