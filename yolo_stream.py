#!/usr/bin/env python3
import os, time, threading, numpy as np, cv2
from picamera2 import Picamera2
from flask import Flask, Response

# ----------------- PARAMETERS (ENV) -----------------
HERE  = os.path.dirname(os.path.abspath(__file__))
MODEL = os.getenv("MODEL", os.path.join(HERE, "yolo.onnx"))
W     = int(os.getenv("W", "320"))     # камера
H     = int(os.getenv("H", "240"))
NET_W = int(os.getenv("NET_W", "640")) # вход сети
NET_H = int(os.getenv("NET_H", "640"))
CONF  = float(os.getenv("CONF", "0.35"))
IOU   = float(os.getenv("IOU",  "0.45"))
PORT  = int(os.getenv("PORT", "8082"))
JQ    = int(os.getenv("JQ",   "70"))   # jpeg качество (0..100)
KEEP  = set(s.strip() for s in os.getenv("KEEP","").split(",") if s.strip()) # пусто = все

# ----------------- COCO CLASSES (built-in) -----------------
COCO = """person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush""".splitlines()
CLASSES = [s.strip() for s in COCO if s.strip()]

# ----------------- NETWORK -----------------
try:
    net = cv2.dnn.readNet(MODEL)
except Exception as e:
    raise SystemExit(f"[ERR] Failed to load model: {MODEL}\n{e}")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ----------------- CAMERA -----------------
cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (W, H), "format": "RGB888"}))
cam.start()

buf = None
cond = threading.Condition()

# ----------------- UTILITIES -----------------
def letterbox(im, new_shape=(640,640), color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    pad_h, pad_w = new_shape[0]-nh, new_shape[1]-nw
    top, bottom = pad_h//2, pad_h - pad_h//2
    left, right = pad_w//2, pad_w - pad_w//2
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (left, top)

def to_blob(img):
    return cv2.dnn.blobFromImage(img, 1/255.0, (NET_W, NET_H), swapRB=True, crop=False)

def postprocess(pred, conf_thr, iou_thr, orig_shape, ratio, dwdh):
    pred = np.squeeze(pred, axis=0)  # (N,85)
    boxes, scores, class_ids = [], [], []
    if pred.ndim != 2 or pred.shape[1] < 85:   # invalid output format
        return boxes, scores, class_ids
    for row in pred:
        obj = row[4]
        if obj < 1e-6: 
            continue
        cls_scores = row[5:]
        cid = int(np.argmax(cls_scores))
        conf = float(obj * cls_scores[cid])
        if conf < conf_thr:
            continue
        if KEEP and (CLASSES[cid] not in KEEP):
            continue
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        x1 = (cx - w/2 - dwdh[0]) / ratio
        y1 = (cy - h/2 - dwdh[1]) / ratio
        x2 = (cx + w/2 - dwdh[0]) / ratio
        y2 = (cy + h/2 - dwdh[1]) / ratio
        x1 = max(0, min(orig_shape[1]-1, int(x1)))
        y1 = max(0, min(orig_shape[0]-1, int(y1)))
        x2 = max(0, min(orig_shape[1]-1, int(x2)))
        y2 = max(0, min(orig_shape[0]-1, int(y2)))
        boxes.append([x1, y1, x2-x1, y2-y1])
        scores.append(conf)
        class_ids.append(cid)
    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thr, iou_thr)
    if len(idxs)==0: 
        return [], [], []
    if isinstance(idxs, (list, np.ndarray)): idxs = np.array(idxs).flatten()
    else: idxs = [int(idxs)]
    return [boxes[i] for i in idxs], [scores[i] for i in idxs], [class_ids[i] for i in idxs]

COLORS = {}
def color_for(cid):
    if cid not in COLORS:
        np.random.seed(cid+123)
        COLORS[cid] = tuple(int(c) for c in np.random.randint(64, 255, 3))
    return COLORS[cid]

def draw_box(img, box, label, conf):
    x,y,w,h = box
    c = color_for(hash(label) % 9999)
    cv2.rectangle(img,(x,y),(x+w,y+h),c,2,cv2.LINE_AA)
    txt = f"{label} {int(conf*100+0.5)}%"
    (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ytxt = max(0, y-6)
    cv2.rectangle(img,(x,ytxt-th-3),(x+tw+4,ytxt+2),c,-1)
    cv2.putText(img,txt,(x+2,ytxt),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

def infer_loop():
    global buf
    t0 = time.time()
    fps = 0.0
    while True:
        frame = cam.capture_array()  # RGB
        letter, ratio, dwdh = letterbox(frame, (NET_H, NET_W))
        blob = to_blob(letter)
        net.setInput(blob)
        out = net.forward()  # (1,N,85)
        boxes, scores, ids = postprocess(out, CONF, IOU, frame.shape, ratio, dwdh)

        annotated = frame.copy()
        for b, s, cid in zip(boxes, scores, ids):
            draw_box(annotated, b, CLASSES[cid], s)

        dt = time.time() - t0
        if dt>0: fps = 1.0/dt
        t0 = time.time()
        cv2.putText(annotated, f"{fps:4.1f} FPS", (5, H-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        ok, jpg = cv2.imencode(".jpg", annotated[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), JQ])
        if ok:
            with cond:
                buf = jpg.tobytes()
                cond.notify_all()

threading.Thread(target=infer_loop, daemon=True).start()

# ----------------- HTTP -----------------
app = Flask(__name__)
@app.route("/")
def index():
    return ('<img src="/stream" style="width:100vw;height:100vh;object-fit:contain;">',200)

@app.route("/stream")
def stream():
    def gen():
        boundary = b"--frame"
        while True:
            with cond:
                cond.wait()
                b = buf
            yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + \
                  str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n"
    return Response(gen(), headers={"Content-Type":"multipart/x-mixed-replace; boundary=frame"})

if __name__ == "__main__":
    print(f"Starting HTTP server at http://0.0.0.0:{PORT}   cam={W}x{H}   net={NET_W}x{NET_H}   CONF={CONF}  IOU={IOU}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
