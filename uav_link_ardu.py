#!/usr/bin/env python3
import argparse
import glob
import sys
import time
from datetime import datetime
from typing import Optional

from pymavlink import mavutil

# ---------- Helpers ----------
def find_serial_port(preferred: Optional[str]) -> Optional[str]:
    if preferred:
        return preferred
    # Typical USB CDC/FTDI device names
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    return candidates[0] if candidates else None

def try_connect(device: str, bauds=(921600, 576000, 230400, 115200, 57600)):
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
                # Close half-open file descriptors between attempts
                m.close()
            except Exception:
                pass
    raise RuntimeError(f"Failed to connect to {device} at common baud rates. Last error: {last_exc}")

def is_mavlink2(master: mavutil.mavfile) -> bool:
    # Heuristic: if link is speaking MAVLink2, master.mav.DIALECT.version==2 or flag on messages
    # pymavlink exposes supported versions via 'master.mav.WIRE_PROTOCOL' but safest is to try.
    return True  # MESSAGE_INTERVAL works on most modern FCs; we still fallback on failure.

def request_message_intervals(master: mavutil.mavfile, hz_map: dict):
    """
    Prefer MAV_CMD_SET_MESSAGE_INTERVAL (MAVLink 2). Fall back to REQUEST_DATA_STREAM if it fails.
    """
    use_message_interval = is_mavlink2(master)
    if use_message_interval:
        try:
            for msg_name, hz in hz_map.items():
                msg_id = master.mavlink.get_msgId(msg_name)
                interval_us = int(1_000_000 / hz) if hz > 0 else -1
                master.mav.command_long_send(
                    master.target_system,
                    master.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    msg_id,          # param1: message ID
                    interval_us,     # param2: interval in microseconds; -1 to stop
                    0, 0, 0, 0, 0
                )
            print("[stream] Set MESSAGE_INTERVALs successfully (MAVLink 2).")
            return
        except Exception as e:
            print(f"[stream] MESSAGE_INTERVAL failed ({e}); falling back to REQUEST_DATA_STREAM…")

    # Fallback: REQUEST_DATA_STREAM (MAV 1.x style)
    try:
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10,  # 10 Hz aggregate
            1    # start
        )
        print("[stream] Requested MAV_DATA_STREAM_ALL @ 10 Hz (fallback).")
    except Exception as e:
        print(f"[stream] REQUEST_DATA_STREAM failed: {e}")

def send_gcs_heartbeat(master: mavutil.mavfile):
    # Identify as a GCS
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0, 0, mavutil.mavlink.MAV_STATE_ACTIVE
    )

def set_mode(master: mavutil.mavfile, mode_str: str):
    # Example: 'GUIDED', 'LOITER', 'AUTO'. Make sure FC supports it.
    master.set_mode(mode_str)
    print(f"[mode] Requested mode: {mode_str}")

def arm_disarm(master: mavutil.mavfile, arm: bool):
    master.arducopter_arm() if arm else master.arducopter_disarm()
    print(f"[arm] Requested {'ARM' if arm else 'DISARM'}")

def fmt_none(x):
    return "—" if x is None else x

# ---------- Main loop ----------
def main():
    ap = argparse.ArgumentParser(description="Pi4 ↔ Flight Controller (USB) MAVLink link using pymavlink")
    ap.add_argument("--port", help="Serial device (e.g. /dev/ttyACM0). Auto-detects if omitted.")
    ap.add_argument("--baud", type=int, help="Baud rate. If omitted, try common speeds.")
    ap.add_argument("--print-every", type=float, default=1.0, help="Status print period (s).")
    ap.add_argument("--emit-heartbeat", action="store_true", help="Send GCS heartbeat @1 Hz.")
    ap.add_argument("--set-mode", help="Set flight mode on start (e.g. GUIDED). USE WITH CARE.")
    ap.add_argument("--arm", action="store_true", help="Arm on start (DANGEROUS).")
    args = ap.parse_args()

    device = find_serial_port(args.port)
    if not device:
        print("No USB serial device found. Plug in FC or pass --port /dev/ttyACM0")
        sys.exit(1)

    if args.baud:
        print(f"[connect] Using explicit {device} @ {args.baud}")
        master = mavutil.mavlink_connection(device, baud=args.baud, autoreconnect=True, source_system=255)
        print("[connect] Waiting for heartbeat (10s timeout)…")
        master.wait_heartbeat(timeout=10)
        baud_used = args.baud
    else:
        master, baud_used = try_connect(device)

    # Configure stream/message intervals
    hz = {
        # name: Hz
        "HEARTBEAT": 1,
        "SYS_STATUS": 1,
        "BATTERY_STATUS": 1,
        "GPS_RAW_INT": 5,
        "GLOBAL_POSITION_INT": 5,
        "ATTITUDE": 10,
        "VFR_HUD": 2,
        "HOME_POSITION": 0.2,
    }
    request_message_intervals(master, hz)

    # Optional (DANGEROUS): change mode / arm. Left commented by default.
    if args.set_mode:
        try:
            set_mode(master, args.set_mode)
        except Exception as e:
            print(f"[mode] Failed: {e}")

    if args.arm:
        print("!!! WARNING: Arming motors on a bench is dangerous. Ensure props off and area is safe. !!!")
        time.sleep(1.5)
        try:
            arm_disarm(master, True)
        except Exception as e:
            print(f"[arm] Failed: {e}")

    last_print = 0.0
    last_hb = 0.0

    print(f"[ok] Connected on {device} @ {baud_used}. Press Ctrl-C to exit.")
    try:
        while True:
            now = time.time()

            # Optional: send our GCS heartbeat
            if args.emit_heartbeat and (now - last_hb) >= 1.0:
                send_gcs_heartbeat(master)
                last_hb = now

            # Non-blocking read; process any incoming message to keep link fresh
            msg = master.recv_msg()

            # Periodic status line
            if (now - last_print) >= args.print_every:
                last_print = now

                gps = master.messages.get("GPS_RAW_INT")
                att = master.messages.get("ATTITUDE")
                vfr = master.messages.get("VFR_HUD")
                bat = master.messages.get("BATTERY_STATUS") or master.messages.get("SYS_STATUS")
                gpos = master.messages.get("GLOBAL_POSITION_INT")

                # Parse fields safely
                if gps:
                    lat = gps.lat / 1e7
                    lon = gps.lon / 1e7
                    alt_msl = gps.alt / 1000.0
                    hdop = getattr(gps, "eph", None)
                    sats = getattr(gps, "satellites_visible", None)
                elif gpos:
                    lat = gpos.lat / 1e7
                    lon = gpos.lon / 1e7
                    alt_msl = gpos.alt / 1000.0
                    hdop = None
                    sats = None
                else:
                    lat = lon = alt_msl = hdop = sats = None

                roll = att.roll if att else None
                pitch = att.pitch if att else None
                yaw = att.yaw if att else None

                airspeed = getattr(vfr, "airspeed", None) if vfr else None
                groundspeed = getattr(vfr, "groundspeed", None) if vfr else None
                throttle = getattr(vfr, "throttle", None) if vfr else None
                alt_rel = getattr(vfr, "alt", None) if vfr else None

                if bat and hasattr(bat, "voltages") and bat.voltages:
                    # BATTERY_STATUS.voltages is array in mV; pick the first non-65535
                    vs = [v for v in bat.voltages if v not in (None, 0, 65535)]
                    voltage = (sum(vs) / len(vs)) / 1000.0 if vs else None
                else:
                    voltage = getattr(master.messages.get("SYS_STATUS"), "voltage_battery", None)
                    voltage = voltage / 1000.0 if voltage else None

                link = master.messages.get("HEARTBEAT")
                base = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{base}] "
                    f"HB sys:{master.target_system} comp:{master.target_component} | "
                    f"GPS {fmt_none(lat):>8}, {fmt_none(lon):>9} alt {fmt_none(round(alt_msl,1) if alt_msl is not None else None)} m "
                    f"(sats {fmt_none(sats)} hdop {fmt_none(hdop)}) | "
                    f"att r/p/y {fmt_none(round(roll,2) if roll is not None else None)}/"
                    f"{fmt_none(round(pitch,2) if pitch is not None else None)}/"
                    f"{fmt_none(round(yaw,2) if yaw is not None else None)} | "
                    f"gs {fmt_none(round(groundspeed,2) if groundspeed is not None else None)} m/s thr {fmt_none(throttle)} | "
                    f"V {fmt_none(round(voltage,2) if voltage is not None else None)}"
                )

            # Brief nap to keep CPU use reasonable
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[exit] Ctrl-C received. Cleaning up…")
    finally:
        try:
            # If we armed in this session, disarm on exit
            if args.arm:
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
