import cv2
import time
import csv
import os
import datetime
from ultralytics import YOLO

model = YOLO("best.pt")

ALERT_COOLDOWN = 10  # seconds
last_alert_time = 0

LOG_FILE = "incident_logs.csv"
ALERT_FOLDER = "static/alerts"

# Create folders/files if missing
os.makedirs(ALERT_FOLDER, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Alert"])

def save_incident(alert_type):
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"LOG SAVED -> {time_now} : {alert_type}")

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time_now, alert_type])

def send_alert(alert_message):
    print(f"🚨 ALERT: {alert_message}")

def generate_frames():
    global last_alert_time

    # Webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return

    threshold = 20  # crowd density limit

    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model(frame)

        person_count = 0
        weapon_detected = False

        for r in results:
            for box in r.boxes:

                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (0, 255, 0)  # default green

                # Count persons
                if label.lower() == "person":
                    person_count += 1

                # Weapon detection
                weapon_classes = [
                    "gun",
                    "knife",
                    "pistol",
                    "handgun",
                    "rifle",
                    "grenade"
                ]

                clean_label = label.lower().strip()

                if any(w in clean_label for w in weapon_classes) and conf > 0.60:

                    weapon_detected = True
                    color = (0, 0, 255)

                    if time.time() - last_alert_time > ALERT_COOLDOWN:

                        filename = f"alert_{int(time.time())}.jpg"
                        cv2.imwrite(f"static/{filename}", frame)

                        save_incident("Weapon Detected")

                        send_alert("Weapon Detected")

                        last_alert_time = time.time()

                # Draw Bounding Box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                text = f"{label} {conf:.2f}"

                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
        last_alert_type = None
        if weapon_detected:
            status = "WEAPON DETECTED"
            status_color = (0, 0, 255)
        
        elif weapon_detected and last_alert_type != "Weapon Detected":
             save_incident("Weapon Detected")
             last_alert_type = "Weapon Detected"

        elif person_count > threshold:

            status = "HIGH CROWD DENSITY"
            status_color = (0, 0, 255)

            if time.time() - last_alert_time > ALERT_COOLDOWN:

                save_incident("High Crowd Density")

                send_alert(
                    f"High Crowd Density Detected ({person_count} people)"
                )

                last_alert_time = time.time()

        else:
            status = "NORMAL"
            status_color = (0, 255, 0)

        cv2.putText(
            frame,
            f"STATUS: {status}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        cv2.putText(
            frame,
            f"People Count: {person_count}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        timestamp = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        cv2.putText(
            frame,
            timestamp,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Camera 01",
            (frame.shape[1] - 180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        ret, buffer = cv2.imencode(".jpg", frame)

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

    cap.release()