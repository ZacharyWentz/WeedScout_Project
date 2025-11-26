# Combined GPS + Hailo Object Detection with LEDs and Continuous Saving
# Created by Peter Russchenberg on 11/10/2025
# Updated 11/25/2025 — Added confidence display/threshold, updated CSV logging, and cleaned up object ID logic

from pathlib import Path
import os
import cv2
import hailo
import atexit
from datetime import datetime
import pytz
import threading
import time
import csv
import math
import queue
import RPi.GPIO as GPIO
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp
from gi.repository import Gst
from pymavlink import mavutil
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# GPIO LED Setup
LED_STARTUP = 23
LED_RUNNING = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_STARTUP, GPIO.OUT)
GPIO.setup(LED_RUNNING, GPIO.OUT)

for _ in range(5):
    GPIO.output(LED_STARTUP, GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(LED_STARTUP, GPIO.LOW)
    time.sleep(0.2)

led_running_active = True
def running_led_thread():
    global led_running_active
    while led_running_active:
        GPIO.output(LED_RUNNING, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(LED_RUNNING, GPIO.LOW)
        time.sleep(0.5)

# GPS Class
class LatestGPS:
    def __init__(self):
        self.lock = threading.Lock()
        self.lat = None
        self.lon = None
        self.alt = None
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def update_position(self, msg):
        with self.lock:
            try:
                self.lat = msg.lat / 1e7
                self.lon = msg.lon / 1e7
                self.alt = msg.relative_alt / 1000.0
            except Exception:
                logging.exception("Failed to parse GLOBAL_POSITION_INT")

    def update_attitude(self, msg):
        with self.lock:
            try:
                r, p, y = msg.roll, msg.pitch, msg.yaw
                if abs(r) <= 6.3 and abs(p) <= 6.3 and abs(y) <= 6.3:
                    self.roll = math.radians(r)
                    self.pitch = math.radians(p)
                    self.yaw = math.radians(y)
                else:
                    self.roll = r
                    self.pitch = p
                    self.yaw = y
            except Exception:
                logging.exception("Failed to parse ATTITUDE")

    def get(self):
        with self.lock:
            return self.lat, self.lon, self.alt, self.roll, self.pitch, self.yaw

# MAVLink Listener
def gps_listener(gps_obj, connection_string="/dev/ttyAMA0"):
    try:
        master = mavutil.mavlink_connection(connection_string)
        logging.info(f"Connecting to MAVLink at {connection_string}...")
        master.wait_heartbeat(timeout=2)
        logging.info("GPS listener connected")

        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            1, 1
        )

        while True:
            msg = master.recv_match(type=['GLOBAL_POSITION_INT'], blocking=True, timeout=2)
            if msg:
                gps_obj.update_position(msg)
            att_msg = master.recv_match(type=['ATTITUDE'], blocking=False)
            if att_msg:
                gps_obj.update_attitude(att_msg)
    except Exception:
        logging.info("MAVLink not available. Object GPS will show N/A.")

# Pixel to ground utility
SENSOR_WIDTH_MM = 6.287
SENSOR_HEIGHT_MM = 4.712
LENS_FOCAL_MM = 6.0

def pixel_to_ground(lat, lon, alt, roll, pitch, yaw, x_pixel, y_pixel, W, H):
    try:
        if lat is None or lon is None or alt is None:
            return None, None
        fx = (LENS_FOCAL_MM / SENSOR_WIDTH_MM) * W
        fy = (LENS_FOCAL_MM / SENSOR_HEIGHT_MM) * H
        cx, cy = W/2.0, H/2.0
        x_cam = (x_pixel - cx) / fx
        y_cam = (y_pixel - cy) / fy
        ray_cam = np.array([x_cam, y_cam, 1.0], dtype=float)
        ray_cam /= np.linalg.norm(ray_cam)

        Rx = np.array([[1,0,0],[0,math.cos(roll),-math.sin(roll)],[0,math.sin(roll), math.cos(roll)]])
        Ry = np.array([[math.cos(pitch),0,math.sin(pitch)],[0,1,0],[-math.sin(pitch),0,math.cos(pitch)]])
        Rz = np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw), math.cos(yaw),0],[0,0,1]])
        R = Rz @ Ry @ Rx
        ray_ned = R @ ray_cam

        vz = ray_ned[2]
        t = alt / (abs(vz)+1e-12)
        north_m = ray_ned[0]*t
        east_m = ray_ned[1]*t

        lat_obj = lat + north_m/111320.0
        lon_obj = lon + east_m/(111320.0*math.cos(math.radians(lat))+1e-12)
        return lat_obj, lon_obj
    except Exception:
        return None, None

# User callback class with GPS
class UserAppCallbackWithGPS(app_callback_class):
    TARGET_CLASS = "Cone"
    TARGET_VIDEO_FPS = 20.0

    def __init__(self, gps_obj, mav_master):
        super().__init__()
        self.gps_obj = gps_obj
        self.mav = mav_master
        self.frame_count = 0
        self.detection_queue = queue.Queue(maxsize=100)
        self.snapshot_queue = queue.Queue(maxsize=100)
        self.tracked_objects = {}
        self.next_object_id = 1
        self.redetect_distance_m = 0.5
        self.confidence_threshold = 80.0 
        self.lock = threading.Lock()
        self.video_writer = None

        usb_root = Path("/media")
        drives = []
        try:
            for sub in usb_root.iterdir():
                if not sub.is_dir(): continue
                for mount in sub.iterdir():
                    if mount.is_dir() and mount.name != "System Volume Information":
                        drives.append(mount)
        except Exception:
            drives = []

        self.base_dir = drives[0] / "WEEDSCOUT" if drives else Path.home() / "Desktop" / "WEEDSCOUT"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(pytz.timezone("US/Central"))
        self.date_str = now.strftime("%Y-%m-%d")
        self.date_folder = self.base_dir / self.date_str
        self.date_folder.mkdir(parents=True, exist_ok=True)

        self.time_str = now.strftime("%H-%M-%S")
        self.output_dir = self.date_folder / f"{self.time_str}_run"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        base_csv_name = f"Weedscout_{self.date_str}"
        self.csv_file = self.output_dir / f"{base_csv_name}.csv"
        counter = 1
        while self.csv_file.exists():
            self.csv_file = self.output_dir / f"{base_csv_name}_{counter}.csv"
            counter += 1
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID#", "Timestamp", "ObjLat", "ObjLon", "Confidence%"])

# Detection callback
preview_frame = None
preview_running = True
video_queue = queue.Queue(maxsize=200)

def app_callback(pad, info, user_data: UserAppCallbackWithGPS):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.frame_count += 1
    try:
        fmt, W, H = get_caps_from_pad(pad)
        frame = get_numpy_from_buffer(buffer, fmt, W, H)
    except Exception:
        return Gst.PadProbeReturn.OK

    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    except Exception:
        detections = []

    try:
        user_data.detection_queue.put_nowait({
            "frame": frame.copy(),
            "detections": detections,
            "frame_count": user_data.frame_count
        })
    except queue.Full:
        pass

    return Gst.PadProbeReturn.OK

# Detection processing thread
def detection_processing_thread(user_data):
    global preview_frame
    while preview_running or not user_data.detection_queue.empty():
        try:
            item = user_data.detection_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        frame = item["frame"]
        frame_count = item["frame_count"]
        detections = item["detections"]

        lat, lon, alt, roll, pitch, yaw = user_data.gps_obj.get()
        frame_overlay = frame.copy()

        # Filter detections by target class and confidence threshold
        frame_targets = [
            d for d in detections
            if d.get_label().lower() == user_data.TARGET_CLASS.lower()
            and (d.get_confidence() * 100.0) >= user_data.confidence_threshold
        ]

        # Top-left overlay info
        top_left_y = 25
        cv2.putText(frame_overlay,
                    f"Time: {datetime.now(pytz.timezone('US/Central')).strftime('%H:%M:%S')} | Objects detected: {len(frame_targets)}",
                    (10, top_left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_overlay,
                    f"Drone: Lat {lat if lat else 'N/A'}, Lon {lon if lon else 'N/A'}, Alt {alt if alt else 'N/A'} m",
                    (10, top_left_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for det in frame_targets:
            bbox = det.get_bbox()
            x_min = int(bbox.xmin() * frame.shape[1])
            y_min = int(bbox.ymin() * frame.shape[0])
            x_max = int(bbox.xmax() * frame.shape[1])
            y_max = int(bbox.ymax() * frame.shape[0])
            cv2.rectangle(frame_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            x_c = (x_min + x_max) / 2.0
            y_c = (y_min + y_max) / 2.0
            obj_lat, obj_lon = pixel_to_ground(lat, lon, alt, roll, pitch, yaw, x_c, y_c, frame.shape[1], frame.shape[0])
            lat_str = f"{obj_lat:.6f}" if obj_lat is not None else "N/A"
            lon_str = f"{obj_lon:.6f}" if obj_lon is not None else "N/A"
            conf_str = f"{det.get_confidence()*100:.1f}%"

            matched_id = None
            with user_data.lock:
                for oid, tracked in user_data.tracked_objects.items():
                    if tracked.get('lat') is None or obj_lat is None:
                        continue
                    if math.hypot(obj_lat - tracked['lat'], obj_lon - tracked['lon']) < user_data.redetect_distance_m:
                        matched_id = oid
                        tracked['bbox'] = (x_min, y_min, x_max, y_max)
                        tracked['last_seen'] = time.time()
                        tracked['lat'] = obj_lat
                        tracked['lon'] = obj_lon
                        break

                if matched_id is None:
                    matched_id = user_data.next_object_id
                    user_data.next_object_id += 1
                    user_data.tracked_objects[matched_id] = {
                        'bbox': (x_min, y_min, x_max, y_max),
                        'last_seen': time.time(),
                        'lat': obj_lat,
                        'lon': obj_lon,
                        'snapshot_taken': False
                    }

                tracked = user_data.tracked_objects[matched_id]

                if not tracked['snapshot_taken']:
                    snapshot_frame = frame_overlay.copy()

                    # Bounding box info
                    cv2.putText(snapshot_frame, f"ID {matched_id} | Lat {lat_str} Lon {lon_str}",
                                (x_min, max(20, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(snapshot_frame, f"Confidence: {conf_str}",
                                (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Top-left overlay
                    cv2.putText(snapshot_frame,
                                f"Time: {datetime.now(pytz.timezone('US/Central')).strftime('%H:%M:%S')} | Objects detected: {len(frame_targets)}",
                                (10, top_left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(snapshot_frame,
                                f"Drone: Lat {lat if lat else 'N/A'}, Lon {lon if lon else 'N/A'}, Alt {alt if alt else 'N/A'} m",
                                (10, top_left_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    try:
                        user_data.snapshot_queue.put_nowait({
                            "frame": snapshot_frame,
                            "filename": f"ID_{matched_id}_f{frame_count}_{int(time.time())}.jpg",
                            "obj_id": matched_id,
                            "frame_count": frame_count,
                            "timestamp": datetime.now(pytz.timezone('US/Central')),
                            "obj_lat": obj_lat,
                            "obj_lon": obj_lon,
                            "confidence": det.get_confidence()*100,
                        })
                        tracked['snapshot_taken'] = True
                    except queue.Full:
                        pass

            # Overlay on live frame
            cv2.putText(frame_overlay, f"ID {matched_id} | Lat {lat_str} Lon {lon_str}",
                        (x_min, max(20, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame_overlay, f"Confidence: {conf_str}",
                        (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        preview_frame = frame_overlay.copy()
        try:
            video_queue.put_nowait(frame_overlay.copy())
        except queue.Full:
            pass

        user_data.detection_queue.task_done()

# Video writer thread
def video_writer_thread(user_data):
    first_frame_obtained = False
    while preview_running or not video_queue.empty():
        try:
            frame_overlay = video_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        try:
            if not first_frame_obtained:
                h, w = frame_overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                user_data.video_writer = cv2.VideoWriter(
                    str(user_data.output_dir / "video.avi"),
                    fourcc,
                    user_data.TARGET_VIDEO_FPS,
                    (w, h)
                )
                first_frame_obtained = True
            user_data.video_writer.write(cv2.cvtColor(frame_overlay, cv2.COLOR_RGB2BGR))
        except Exception:
            logging.exception("Error writing video frame")
        finally:
            video_queue.task_done()

# Snapshot writer thread
def snapshot_writer_thread(user_data):
    while preview_running or not user_data.snapshot_queue.empty():
        try:
            item = user_data.snapshot_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        snapshot_file = user_data.output_dir / item['filename']
        try:
            cv2.imwrite(str(snapshot_file), cv2.cvtColor(item['frame'], cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            with open(user_data.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    item['obj_id'],
                    item['timestamp'].strftime("%H:%M:%S"),
                    item.get('obj_lat',"N/A"),
                    item.get('obj_lon',"N/A"),
                    item.get('confidence', "N/A")
                ])
        except Exception:
            logging.exception(f"Failed to save snapshot {snapshot_file}")
        user_data.snapshot_queue.task_done()

# Preview thread
#def preview_thread():
    #global preview_frame, preview_running
    #while preview_running:
        #if preview_frame is not None:
            #try:
                #cv2.imshow("Preview", cv2.cvtColor(preview_frame, cv2.COLOR_RGB2BGR))
            #except Exception:
                #pass
        #if cv2.waitKey(1) & 0xFF == 27:
            #preview_running = False
            #try: Gst.main_quit()
            #except: pass
            #break
        #time.sleep(0.01)

# Cleanup
def cleanup(user_data):
    global led_running_active, preview_running
    led_running_active = False
    preview_running = False

    logging.info("Cleaning up, flushing queues...")

    while not user_data.detection_queue.empty():
        time.sleep(0.05)
    while not video_queue.empty():
        time.sleep(0.05)
    while not user_data.snapshot_queue.empty():
        time.sleep(0.05)

    try:
        if user_data.video_writer is not None:
            user_data.video_writer.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        GPIO.cleanup()
    except Exception:
        pass

    logging.info("Cleanup complete.")

# Main
if __name__=="__main__":
    project_root = Path(__file__).resolve().parent.parent
    os.environ["HAILO_ENV_FILE"] = str(project_root / ".env")

    gps_obj = LatestGPS()
    threading.Thread(target=gps_listener, args=(gps_obj,), daemon=True).start()
    threading.Thread(target=running_led_thread, daemon=True).start()

    user_callback = UserAppCallbackWithGPS(gps_obj, None)
    threading.Thread(target=detection_processing_thread, args=(user_callback,), daemon=True).start()
    threading.Thread(target=video_writer_thread, args=(user_callback,), daemon=True).start()
    threading.Thread(target=snapshot_writer_thread, args=(user_callback,), daemon=True).start()
    #threading.Thread(target=preview_thread, daemon=True).start()

    atexit.register(cleanup, user_callback)

    app = GStreamerDetectionApp(
        app_callback=app_callback,
        user_data=user_callback
    )
    try:
        app.run()
    except KeyboardInterrupt:
        cleanup(user_callback)
