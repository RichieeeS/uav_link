import sys
import subprocess
required = [
    'prompt_toolkit', 'pymavlink', 'flask', 'numpy', 'opencv-python', 'ultralytics', 'onnx', 'onnxruntime', 'onnxslim'
]
missing = []
for m in required:
    try:
        __import__(m.split('-')[0])
    except ImportError:
        missing.append(m)
if missing:
    print(f"Installing missing modules: {', '.join(missing)}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--break-system-packages'] + missing)
#!/usr/bin/env python3
import argparse
import glob
import sys
import time
import select
from prompt_toolkit import prompt
from prompt_toolkit.patch_stdout import patch_stdout
import os
from datetime import datetime
from typing import Optional

from pymavlink import mavutil

ARDUPILOT_MODES = {
    "STABILIZE": 0,
    "ACRO": 1, 
    "ALT_HOLD": 2,
    "AUTO": 3,
    "GUIDED": 4,
    "LOITER": 5,
    "RTL": 6,
    "CIRCLE": 7,
    "LAND": 9,
    "DRIFT": 11,
    "SPORT": 13,
    "FLIP": 14,
    "AUTOTUNE": 15,
    "POSHOLD": 16,
    "BRAKE": 17,
    "THROW": 18,
    "AVOID_ADSB": 19,
    "GUIDED_NOGPS": 20,
    "SMART_RTL": 21,
    "FLOWHOLD": 22,
    "FOLLOW": 23,
    "ZIGZAG": 24,
    "SYSTEMID": 25,
    "AUTOROTATE": 26,
    "AUTO_RTL": 27
}

def find_serial_port(preferred: Optional[str]) -> Optional[str]:
    if preferred:
        return preferred
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    return candidates[0] if candidates else None

def try_connect(device: str, bauds=(115200, 57600)):
    last_exc = None
    for baud in bauds:
        try:
            print(f"[connect] Trying {device} @ {baud} …")
            m = mavutil.mavlink_connection(device, baud=baud, autoreconnect=True, source_system=255)
            print("[connect] Waiting for heartbeat (10s timeout)…")
            m.wait_heartbeat(timeout=10)
            print(f"[connect] Heartbeat from sys:{m.target_system} comp:{m.target_component} @ {baud}")
            return m, baud
        except Exception as e:
            last_exc = e
            try:
                m.close()
            except Exception:
                pass
    raise RuntimeError(f"Failed to connect to {device}. Last error: {last_exc}")

def request_message_intervals(master: mavutil.mavfile, hz_map: dict):
    if hasattr(master, 'mavlink') and hasattr(master.mavlink, 'get_msgId'):
        try:
            for msg_name, hz in hz_map.items():
                msg_id = master.mavlink.get_msgId(msg_name)
                interval_us = int(1_000_000 / hz) if hz > 0 else -1
                master.mav.command_long_send(
                    master.target_system,
                    master.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    msg_id,
                    interval_us,
                    0, 0, 0, 0, 0
                )
            print("[stream] MESSAGE_INTERVAL requests sent.")
            return
        except Exception as e:
            print(f"[stream] MESSAGE_INTERVAL failed ({e}); falling back to REQUEST_DATA_STREAM…")
    # Fallback
    try:
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10,
            1
        )
        print("[stream] Requested MAV_DATA_STREAM_ALL @ 10 Hz (fallback).")
    except Exception as e:
        print(f"[stream] REQUEST_DATA_STREAM failed: {e}")

def send_gcs_heartbeat(master: mavutil.mavfile):
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0, 0, mavutil.mavlink.MAV_STATE_ACTIVE
    )

def arm_disarm(master: mavutil.mavfile, arm: bool):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1 if arm else 0,
        0, 0, 0, 0, 0, 0
    )
    print(f"[arm] Sent {'ARM' if arm else 'DISARM'} command")

def set_mode_ardupilot(master: mavutil.mavfile, mode_name: str):
    mode_name = mode_name.upper()
    if mode_name not in ARDUPILOT_MODES:
        print(f"[mode] Unknown mode '{mode_name}'. Valid: {list(ARDUPILOT_MODES.keys())}")
        return
    mode_id = ARDUPILOT_MODES[mode_name]
    # Set mode using ArduPilot convention
    master.set_mode(mode_id)
    print(f"[mode] Requested ArduPilot mode {mode_name} ({mode_id})")

def fmt_none(x):
    return "—" if x is None else x

def main():
    import subprocess
    import signal
    import re
    # Check for other processes using the camera (e.g., /dev/video0)
    try:
        lsof_out = subprocess.check_output(['lsof', '/dev/video0'], stderr=subprocess.DEVNULL).decode()
        pids = set()
        for line in lsof_out.splitlines()[1:]:
            m = re.match(r'\S+\s+(\d+)', line)
            if m:
                pids.add(int(m.group(1)))
        for pid in pids:
            if pid != os.getpid():
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
        if pids:
            print(f"Killed processes using /dev/video0: {', '.join(map(str, pids))}")
    except Exception:
        pass
    log_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'telemetry_{log_id}.log'
    log_file = open(log_filename, 'a')
    print(f"[log] Telemetry will be saved to {log_filename}")

    import subprocess
    import signal
    yolo_proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), 'yolo_stream.py')])
    ap = argparse.ArgumentParser(description="Raspberry Pi ↔ ArduPilot FC (USB) MAVLink link using pymavlink")
    ap.add_argument("--port", help="Serial device (e.g. /dev/ttyACM0). Auto-detects if omitted.")
    ap.add_argument("--baud", type=int, help="Baud rate. If omitted, tries common speeds.")
    ap.add_argument("--print-every", type=float, default=1.0, help="Status print period (s).")
    ap.add_argument("--emit-heartbeat", action="store_true", help="Send GCS heartbeat @1 Hz.")
    ap.add_argument("--set-mode", help=f"Set ArduPilot flight mode on start. Options: {list(ARDUPILOT_MODES.keys())}")
    ap.add_argument("--arm", action="store_true", help="Arm on start (dangerous!).")
    args = ap.parse_args()

    device = find_serial_port(args.port)

    if not device:
        print("No USB serial device found. Plug in FC or pass --port /dev/ttyACM0")
        sys.exit(1)

    if args.baud:
        print(f"[connect] Using explicit {device} @ {args.baud}")
        master = mavutil.mavlink_connection(device, baud=args.baud, autoreconnect=True, source_system=255)
        master.wait_heartbeat(timeout=10)
        baud_used = args.baud
    else:
        master, baud_used = try_connect(device)

    hz = {
        "HEARTBEAT": 1,
        "SYS_STATUS": 1,
        "BATTERY_STATUS": 1,
        "GPS_RAW_INT": 5,
        "GLOBAL_POSITION_INT": 5,
        "ATTITUDE": 10,
        "VFR_HUD": 2
    }
    request_message_intervals(master, hz)


    if args.set_mode:
        set_mode_ardupilot(master, args.set_mode)
        if args.arm:
            print("!!! WARNING: Arming motors on not a clear surface is dangerous. Remove obstacles first. !!!")
            time.sleep(1.5)
            arm_disarm(master, True)

    print(f"[ok] Connected on {device} @ {baud_used}. Type 'help' for commands. Press Ctrl-C or type 'exit' to quit.")
    last_print = 0.0
    last_hb = 0.0
    latest = {}
    import threading
    import queue

    stop_event = threading.Event()
    input_queue = queue.Queue()


    def telemetry_loop():
        nonlocal last_print, last_hb, latest
        while not stop_event.is_set():
            try:
                now = time.time()
                if args.emit_heartbeat and (now - last_hb) >= 1.0:
                    send_gcs_heartbeat(master)
                    last_hb = now
                msg = master.recv_msg()
                if msg:
                    latest[msg.get_type()] = msg
                if (now - last_print) >= args.print_every:
                    last_print = now
                    gps = latest.get("GPS_RAW_INT")
                    att = latest.get("ATTITUDE")
                    vfr = latest.get("VFR_HUD")
                    bat = latest.get("BATTERY_STATUS") or latest.get("SYS_STATUS")
                    gpos = latest.get("GLOBAL_POSITION_INT")
                    if gps:
                        lat = gps.lat / 1e7
                        lon = gps.lon / 1e7
                        alt_msl = gps.alt / 1000.0
                        sats = getattr(gps, "satellites_visible", None)
                    elif gpos:
                        lat = gpos.lat / 1e7
                        lon = gpos.lon / 1e7
                        alt_msl = gpos.alt / 1000.0
                        sats = None
                    else:
                        lat = lon = alt_msl = sats = None
                    roll = att.roll if att else None
                    pitch = att.pitch if att else None
                    yaw = att.yaw if att else None
                    groundspeed = getattr(vfr, "groundspeed", None) if vfr else None
                    throttle = getattr(vfr, "throttle", None) if vfr else None
                    if bat and hasattr(bat, "voltages") and bat.voltages:
                        vs = [v for v in bat.voltages if v not in (None, 0, 65535)]
                        voltage = (sum(vs) / len(vs)) / 1000.0 if vs else None
                    else:
                        voltage = None
                    base = datetime.now().strftime("%H:%M:%S")
                    latest['status_str'] = (
                        f"[{base}] "
                        f"GPS {fmt_none(lat)}, {fmt_none(lon)} alt {fmt_none(round(alt_msl,1) if alt_msl else None)}m "
                        f"(sats {fmt_none(sats)}) | "
                        f"att r/p/y {fmt_none(round(roll,2) if roll else None)}/"
                        f"{fmt_none(round(pitch,2) if pitch else None)}/"
                        f"{fmt_none(round(yaw,2) if yaw else None)} | "
                        f"gs {fmt_none(round(groundspeed,2) if groundspeed else None)} m/s thr {fmt_none(throttle)} | "
                        f"V {fmt_none(round(voltage,2) if voltage else None)}"
                    )
                    print(latest['status_str'], flush=True)
                    try:
                        log_file.write(latest['status_str'] + '\n')
                        log_file.flush()
                    except Exception:
                        pass
                time.sleep(0.01)
            except TypeError as e:
                if "NoneType" in str(e) and "item assignment" in str(e):
                    continue
                else:
                    print(f"[telemetry_loop] Exception: {e}")
                    continue

    def input_loop():
        try:
            with patch_stdout():
                while not stop_event.is_set():
                    cmd = prompt('> ')
                    input_queue.put(cmd.strip())
        except EOFError:
            stop_event.set()
        except KeyboardInterrupt:
            stop_event.set()

    t1 = threading.Thread(target=telemetry_loop, daemon=True)
    t2 = threading.Thread(target=input_loop, daemon=True)
    t1.start()
    t2.start()

    try:
        while not stop_event.is_set():
            try:
                cmd = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not cmd:
                continue
            try:
                log_file.write(f"[command] {cmd}\n")
                log_file.flush()
            except Exception:
                pass
            parts = cmd.split()
            if not parts:
                continue
            c = parts[0].lower()
            output = None
            if c == 'exit' or c == 'quit':
                output = '[exit] Exiting...'
                print(output)
                stop_event.set()
            elif c == 'help':
                output = 'Commands: mode [MODE], heartbeat, status, exit, help\nValid ArduPilot modes:\n' + ', '.join(ARDUPILOT_MODES.keys())
                print('Commands: mode [MODE], heartbeat, status, exit, help')
                print('Valid ArduPilot modes:')
                print(', '.join(ARDUPILOT_MODES.keys()))
            elif c == 'mode' and len(parts) > 1:
                try:
                    set_mode_ardupilot(master, parts[1])
                    output = f"[mode] Set mode to {parts[1]}"
                except Exception as e:
                    output = f'[mode] Failed: {e}'
                    print(output)
            elif c == 'heartbeat':
                try:
                    send_gcs_heartbeat(master)
                    output = '[heartbeat] Sent GCS heartbeat.'
                    print(output)
                except Exception as e:
                    output = f'[heartbeat] Failed: {e}'
                    print(output)
            elif c == 'status':
                output = latest.get('status_str', '[status] No telemetry yet.')
                print(output)
            else:
                output = '[error] Unknown command. Type help.'
                print(output)
            if output:
                try:
                    log_file.write(f"[output] {output}\n")
                    log_file.flush()
                except Exception:
                    pass
        stop_event.set()
    except KeyboardInterrupt:
        print("\n[exit] Ctrl-C received. Cleaning up…")
        try:
            log_file.write("[output] KeyboardInterrupt: Exiting...\n")
            log_file.flush()
        except Exception:
            pass
        stop_event.set()
    finally:
        if args.arm:
            try:
                arm_disarm(master, False)
            except Exception:
                pass
        try:
            master.close()
        except Exception:
            pass
        try:
            log_file.close()
        except Exception:
            pass
        try:
            yolo_proc.terminate()
            yolo_proc.wait(timeout=5)
        except Exception:
            pass
        print("[exit] Link closed.")

    t1 = threading.Thread(target=telemetry_loop, daemon=True)
    t2 = threading.Thread(target=input_loop, daemon=True)
    t1.start()
    t2.start()

    try:
        while not stop_event.is_set():
            try:
                cmd = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not cmd:
                continue
            parts = cmd.split()
            if not parts:
                continue
            c = parts[0].lower()
            if c == 'exit' or c == 'quit':
                print('[exit] Exiting...')
                stop_event.set()
            elif c == 'help':
                print('Commands: mode [MODE], heartbeat, status, exit, help')
                print('Valid ArduPilot modes:')
                print(', '.join(ARDUPILOT_MODES.keys()))
            elif c == 'mode' and len(parts) > 1:
                try:
                    set_mode_ardupilot(master, parts[1])
                except Exception as e:
                    print(f'[mode] Failed: {e}')
            elif c == 'heartbeat':
                try:
                    send_gcs_heartbeat(master)
                    print('[heartbeat] Sent GCS heartbeat.')
                except Exception as e:
                    print(f'[heartbeat] Failed: {e}')
            elif c == 'status':
                print(latest.get('status_str', '[status] No telemetry yet.'))
            else:
                print('[error] Unknown command. Type help.')
        stop_event.set()
    except KeyboardInterrupt:
        print("\n[exit] Ctrl-C received. Cleaning up…")
        stop_event.set()
    finally:
        if args.arm:
            try:
                arm_disarm(master, False)
            except Exception:
                pass
        try:
            master.close()
        except Exception:
            pass
        print("[exit] Link closed.")

if __name__ == "__main__":
    main()
