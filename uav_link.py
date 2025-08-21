#!/usr/bin/env python3
import argparse
import glob
import sys
import time
from datetime import datetime
from typing import Optional

from pymavlink import mavutil

INAV_MODES = {
    "MANUAL": 0,
    "ANGLE": 1,
    "HORIZON": 2,
    "POSHOLD": 5,   
    "RTH": 6,       
    "WP": 7,        
    "ACRO": 8,
}

def find_serial_port(preferred: Optional[str]) -> Optional[str]:
    if preferred:
        return preferred
    candidates = sorted(glob.glob("/dev/ttyACM1") + glob.glob("/dev/ttyUSB*"))
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
    except Exception as e:
        print(f"[stream] Failed to set message intervals: {e}")

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

def set_mode_inav(master: mavutil.mavfile, mode_name: str):
    if mode_name.upper() not in INAV_MODES:
        print(f"[mode] Unknown mode '{mode_name}'. Valid: {list(INAV_MODES.keys())}")
        return
    custom_mode = INAV_MODES[mode_name.upper()]
    base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        base_mode,
        custom_mode,
        0, 0, 0, 0, 0
    )
    print(f"[mode] Requested iNav mode {mode_name} ({custom_mode})")

def fmt_none(x):
    return "—" if x is None else x

def main():
    ap = argparse.ArgumentParser(description="Raspberry Pi ↔ iNav FC (USB) MAVLink link using pymavlink")
    ap.add_argument("--port", help="Serial device (e.g. /dev/ttyACM0). Auto-detects if omitted.")
    ap.add_argument("--baud", type=int, help="Baud rate. If omitted, tries common speeds.")
    ap.add_argument("--print-every", type=float, default=1.0, help="Status print period (s).")
    ap.add_argument("--emit-heartbeat", action="store_true", help="Send GCS heartbeat @1 Hz.")
    ap.add_argument("--set-mode", help=f"Set iNav flight mode on start. Options: {list(INAV_MODES.keys())}")
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
        "VFR_HUD": 2,
    }
    request_message_intervals(master, hz)

    if args.set_mode:
        set_mode_inav(master, args.set_mode)
    if args.arm:
        print("!!! WARNING: Arming motors on not a clear surface is dangerous. Remove obstacles first. !!!")
        time.sleep(1.5)
        arm_disarm(master, True)

    last_print = 0.0
    last_hb = 0.0

    print(f"[ok] Connected on {device} @ {baud_used}. Press Ctrl-C to exit.")
    try:
        while True:
            now = time.time()

            if args.emit_heartbeat and (now - last_hb) >= 1.0:
                send_gcs_heartbeat(master)
                last_hb = now

            msg = master.recv_msg()

            if (now - last_print) >= args.print_every:
                last_print = now

                gps = master.messages.get("GPS_RAW_INT")
                att = master.messages.get("ATTITUDE")
                vfr = master.messages.get("VFR_HUD")
                bat = master.messages.get("BATTERY_STATUS") or master.messages.get("SYS_STATUS")

                if gps:
                    lat = gps.lat / 1e7
                    lon = gps.lon / 1e7
                    alt_msl = gps.alt / 1000.0
                    sats = getattr(gps, "satellites_visible", None)
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
                print(
                    f"[{base}] "
                    f"GPS {fmt_none(lat)}, {fmt_none(lon)} alt {fmt_none(round(alt_msl,1) if alt_msl else None)}m "
                    f"(sats {fmt_none(sats)}) | "
                    f"att r/p/y {fmt_none(round(roll,2) if roll else None)}/"
                    f"{fmt_none(round(pitch,2) if pitch else None)}/"
                    f"{fmt_none(round(yaw,2) if yaw else None)} | "
                    f"gs {fmt_none(round(groundspeed,2) if groundspeed else None)} m/s thr {fmt_none(throttle)} | "
                    f"V {fmt_none(round(voltage,2) if voltage else None)}"
                )

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[exit] Ctrl-C received. Cleaning up…")
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
