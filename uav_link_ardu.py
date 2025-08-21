def goto_gps(master, lat, lon, alt):
    # Convert to required MAVLink format (degrees * 1e7, meters)
    lat_int = int(float(lat) * 1e7)
    lon_int = int(float(lon) * 1e7)
    alt_m = float(alt)
    master.mav.set_position_target_global_int_send(
        int(time.time()*1e3), # time_boot_ms
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000111111111000, # type_mask: ignore velocity/accel/yaw
        lat_int, lon_int, alt_m,
        0, 0, 0, # vx, vy, vz
        0, 0, 0, # afx, afy, afz
        0, 0 # yaw, yaw_rate
    )
    print(f"[goto] Commanded to fly to lat={lat}, lon={lon}, alt={alt}")
def set_mode(master, mode_name):
    mode_id = master.mode_mapping().get(mode_name.upper())
    if mode_id is None:
        print(f"[mode] Unknown mode: {mode_name}")
        return False
    master.set_mode(mode_id)
    print(f"[mode] Set mode to {mode_name}")
    return True

def arm_vehicle(master):
    master.arducopter_arm()
    print("[arm] Sent arm command")

def disarm_vehicle(master):
    master.arducopter_disarm()
    print("[arm] Sent disarm command")
#!/usr/bin/env python3
import argparse, sys, time, threading, csv, os
from datetime import datetime
import numpy as np
from pymavlink import mavutil


from picamera2 import Picamera2
import onnxruntime as ort
from yolo_nano_stream import letterbox, parse_predictions, COCO, IMG_SIZE


def find_serial_port(preferred=None):
    import glob
    if preferred:
        return preferred
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    return candidates[0] if candidates else None

def try_connect(device, bauds=(921600, 576000, 230400, 115200, 57600)):
    for baud in bauds:
        try:
            print(f"[connect] Trying {device} @ {baud} …")
            m = mavutil.mavlink_connection(device, baud=baud, autoreconnect=True, source_system=255)
            m.wait_heartbeat(timeout=10)
            print(f"[connect] Heartbeat from sys:{m.target_system} comp:{m.target_component} @ {baud}")
            return m
        except Exception:
            try: m.close()
            except: pass
    raise RuntimeError("Failed to connect to FC")

def send_gcs_heartbeat(master):
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0, 0, mavutil.mavlink.MAV_STATE_ACTIVE
    )

def send_yolo_detection(master, det_id, label, conf, x1,y1,x2,y2, img_w, img_h, csv_writer=None):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    master.mav.vision_detection_send(
        det_id,
        int(conf*100),
        0,0,0,
        label.encode("utf-8"),
        int(cx*1000), int(cy*1000), int(w*1000), int(h*1000),
        0
    )
    ts = datetime.now().isoformat()
    print(f"[yolo→mav DET] {label} conf={conf:.2f} center=({cx:.2f},{cy:.2f}) size=({w:.2f},{h:.2f})")
    if csv_writer:
        csv_writer.writerow([ts, "DETECTION", det_id, label, conf, cx, cy, w, h])

def send_vision_position(master, ts_usec, cx, cy, size, csv_writer=None):
    x = (cx - 0.5) * 2.0
    y = (cy - 0.5) * -2.0
    z = max(0.1, 1.0 - size)

    roll = pitch = yaw = 0.0
    master.mav.vision_position_estimate_send(
        ts_usec,
        x, y, z,
        roll, pitch, yaw
    )
    ts = datetime.now().isoformat()
    print(f"[yolo→mav POSE] x={x:.2f} y={y:.2f} z={z:.2f}")
    if csv_writer:
        csv_writer.writerow([ts, "POSE", "", "", "", x, y, z])


def yolo_worker(master, model_path, width, height, csv_writer):
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": (width, height), "format": "RGB888"}))
    cam.start()

    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp  = sess.get_inputs()[0].name
    out  = sess.get_outputs()[0].name

    det_id = 0
    while True:
        frame = cam.capture_array()
        img, r, (dw, dh) = letterbox(frame, (IMG_SIZE, IMG_SIZE))
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2,0,1))[None, ...]
        raw = sess.run([out], {inp: x})[0]
        dets = parse_predictions(raw, frame.shape[:2], r, dw, dh)

        if dets:
            for x1,y1,x2,y2,cls,conf in dets:
                label = COCO[cls]
                send_yolo_detection(master, det_id, label, conf, x1,y1,x2,y2, frame.shape[1], frame.shape[0], csv_writer)
                det_id += 1

            x1,y1,x2,y2,cls,conf = dets[0]
            cx = (x1+x2)/2/frame.shape[1]
            cy = (y1+y2)/2/frame.shape[0]
            size = ((x2-x1)/frame.shape[1] + (y2-y1)/frame.shape[0]) / 2
            ts_usec = int(time.time()*1e6)
            send_vision_position(master, ts_usec, cx, cy, size, csv_writer)

        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description="YOLO camera → ArduPilot MAVLink bridge with logging")
    ap.add_argument("-p", "--port", help="Serial device (/dev/ttyACM0). Auto-detects if omitted.")
    ap.add_argument("-b", "--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    ap.add_argument("-m", "--model", default="/home/asd2/yolo_cam/yolo.onnx", help="YOLO ONNX model path")
    ap.add_argument("-e", "--emit-heartbeat", action="store_true", help="Emit GCS heartbeat")
    ap.add_argument("-l", "--log", default="yolo_detections.csv", help="CSV log file path")
    ap.add_argument("-w", "--cam-width", type=int, default=320, help="Camera width (default: 320)")
    ap.add_argument("-t", "--cam-height", type=int, default=240, help="Camera height (default: 240)")

    ap.add_argument("-a", "--arm", action="store_true", help="Arm the vehicle on startup")
    ap.add_argument("-d", "--disarm", action="store_true", help="Disarm the vehicle on startup")
    ap.add_argument("-M", "--mode", type=str, help="Set flight mode (e.g., GUIDED, AUTO, LOITER)")
    args = ap.parse_args()

    device = find_serial_port(args.port)
    if not device:
        print("No FC found."); sys.exit(1)
    master = try_connect(device, bauds=(args.baud,))

    log_exists = os.path.exists(args.log)
    log_file = open(args.log, "a", newline="")
    csv_writer = csv.writer(log_file)
    if not log_exists:
        csv_writer.writerow(["timestamp","type","id","label","confidence","x","y","z_or_w","h"])  


    
    if args.mode:
        set_mode(master, args.mode)
        time.sleep(1)
    if args.arm:
        arm_vehicle(master)
        time.sleep(1)
    if args.disarm:
        disarm_vehicle(master)
        time.sleep(1)

    threading.Thread(target=yolo_worker, args=(master, args.model, args.cam_width, args.cam_height, csv_writer), daemon=True).start()

    last_hb = 0
    def mavlink_loop():
        while True:
            now = time.time()
            if args.emit_heartbeat and (now - last_hb) >= 1.0:
                send_gcs_heartbeat(master)
                nonlocal last_hb
                last_hb = now
            msg = master.recv_match(blocking=False)
            if msg and msg.get_type().startswith("OSD_"):
                print(f"[osd] {msg}")
            time.sleep(0.05)

    t = threading.Thread(target=mavlink_loop, daemon=True)
    t.start()

    try:
        while True:
            cmd = input("[cmd] > ").strip()
            if cmd == "exit":
                print("[exit] Exiting interactive mode.")
                break
            elif cmd == "arm":
                arm_vehicle(master)
            elif cmd == "disarm":
                disarm_vehicle(master)
            elif cmd.startswith("mode "):
                mode_name = cmd.split(None, 1)[1]
                set_mode(master, mode_name)
            elif cmd.startswith("goto "):
                parts = cmd.split()
                if len(parts) == 4:
                    goto_gps(master, parts[1], parts[2], parts[3])
                else:
                    print("Usage: goto <lat> <lon> <alt>")
            elif cmd == "help":
                print("Commands: arm, disarm, mode <MODE>, goto <lat> <lon> <alt>, exit, help")
            else:
                print("Unknown command. Type 'help' for options.")
    except KeyboardInterrupt:
        print("\n[exit] Ctrl-C")
    finally:
        log_file.close()

if __name__ == "__main__":
    main()
