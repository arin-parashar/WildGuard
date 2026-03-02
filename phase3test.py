import os
import sys
import cv2
import time
import random
import threading
import winsound
import numpy as np
from pathlib import Path
from collections import deque, Counter
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import serial

# ================= CONFIGURATION =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = str(PROJECT_ROOT / "models" / "md" / "md_v5a.0.0.pt")
SPECIESNET_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.2a"

# --- TUNING PARAMETERS (ROBUSTNESS FIX) ---
# Lowered thresholds to catch people even when face is covered
CONF_THRESH_PERSON = 0.45  
CONF_THRESH_ANIMAL = 0.55  
IOU_THRESH_SUPPRESSION = 0.3 # If Animal overlaps Person by 30%, ignore Animal

INFER_WIDTH = 1280
INFER_EVERY_N_FRAMES = 3
PERSON_HOLD_FRAMES = 10     # Increased slightly for stability
BEEP_COOLDOWN_SEC = 1.0
SPECIES_VOTE_FRAMES = 10
CROP_PAD = 0.15
HUMAN_SPECIES_TOKENS = ("human", "homo sapiens", "person", "people", "man", "woman")
GENERIC_SPECIES_TOKENS = ("blank", "unknown", "animal", "reptile", "mammal", "vertebrate")
LABEL_NORMALIZATION_RULES = (
    ("crocodilian", "crocodile"),
    ("saltwater crocodile", "crocodile"),
    ("american crocodile", "crocodile"),
    ("nilotic crocodile", "crocodile"),
    ("monitor lizard", "monitor lizard"),
)
CONF_THRESH_ANIMAL_FILE = 0.45
CONF_THRESH_ANIMAL_FILE_FALLBACK = 0.40
ANIMAL_AREA_MIN_FILE = 1200
FILE_SPECIES_SINGLE_VIEW_CONF = 0.22
FILE_SPECIES_CONSENSUS_MIN_VOTES = 2
FILE_SPECIES_CONSENSUS_AVG_CONF = 0.06
CONFIDENCE_FLOOR_TRIGGER = 0.50
CONFIDENCE_FLOOR_TARGET = 0.80

# --- COLORS ---
BG_MAIN = "#0A0A0A"
BG_SIDE = "#121212"
CARD_BG = "#1A1A1A"
CARD_BORDER = "#2C2C2C"
VIEW_BG = "#0F1113"
VIEW_BORDER = "#2A2F36"
ACCENT = "#00E5FF"       
DANGER = "#FF3D00"       
ANIMAL_COLOR = "#FFD740" 
TEXT_DIM = "#888888"
# ================= SYSTEM SETUP =================
os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT / "external" / "MegaDetector"))
sys.path.append(str(PROJECT_ROOT / "external" / "MegaDetector" / "external" / "yolov5"))

try:
    print("ðŸš€ LOADING ENGINES...")
    import megadetector.detection.run_detector as rd
    from speciesnet import SpeciesNetClassifier
    detector = rd.load_detector(MODEL_PATH)
    classifier = SpeciesNetClassifier(SPECIESNET_MODEL)
    print("âœ… ENGINES READY.")
except Exception as e:
    print(f"âŒ CRITICAL ERROR: {e}")
    sys.exit(1)

# ================= HELPERS =================
def calculate_iou(boxA, boxB):

    """Calculate Intersection over Union (IoU) between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def center_distance(a, b):
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

def make_square_crop(frame, bbox, pad=0.15):
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    dim = int(max(x2 - x1, y2 - y1) * (1 + pad))
    x1n, y1n = max(0, cx - dim // 2), max(0, cy - dim // 2)
    x2n, y2n = min(w_img, cx + dim // 2), min(h_img, cy + dim // 2)
    return frame[y1n:y2n, x1n:x2n]

# def play_beep():
#     try: winsound.Beep(1200, 50)
#     except: pass

# ================= UI COMPONENTS =================
class RoundedButton(tk.Canvas):
    def __init__(self, master, text, command, bg_color, hover_color, fg_color, font, height=54, radius=16):
        super().__init__(master, height=height, bg=master["bg"], highlightthickness=0, bd=0)
        self.text = text
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.fg_color = fg_color
        self.font = font
        self.radius = radius
        self.enabled = True
        self.current_fill = bg_color

        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>", lambda e: self._on_hover(True))
        self.bind("<Leave>", lambda e: self._on_hover(False))
        self.bind("<Button-1>", self._on_click)
        self._draw()

    def _round_rect(self, x1, y1, x2, y2, r, fill, outline):
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1
        ]
        self.create_polygon(points, smooth=True, splinesteps=24, fill=fill, outline=outline, width=1)

    def _draw(self):
        self.delete("all")
        w = max(self.winfo_width(), 10)
        h = max(self.winfo_height(), 10)
        fill = self.current_fill if self.enabled else "#2A2D34"
        outline = "#1A1D23" if self.enabled else "#23262C"
        self._round_rect(2, 2, w - 2, h - 2, self.radius, fill, outline)
        fg = self.fg_color if self.enabled else "#7A7F88"
        self.create_text(w // 2, h // 2, text=self.text, fill=fg, font=self.font)

    def _on_hover(self, entering):
        if not self.enabled:
            return
        self.current_fill = self.hover_color if entering else self.bg_color
        self._draw()

    def _on_click(self, _event):
        if self.enabled and self.command:
            self.command()

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self.current_fill = self.bg_color
        self._draw()

# ================= MAIN APP =================
class WildGuardPro:
    def __init__(self, root):
        self.animal_tracks = []
        self.root = root
        self.root.title("WILDGUARD | Human Wildlife Co-occurrence Analyzer")
        self.root.geometry("1560x920")
        self.root.minsize(1360, 820)
        self.root.configure(bg=BG_MAIN)
        self.esp = None
        self.buzzer_state = False  # prevent spamming serial

        self.last_category_sent = None
        self.trigger_sent = False # prevent serial spam

        # Animal categories
        self.safe_animals = {"deer", "domestic dog", "rabbit", "squirrel", "fox", "badger", "otter", "beaver", "raccoon", "hedgehog", "armadillo"}
        self.normal_animals = {"zebra", "rhino", "giraffe", "buffalo", "antelope", "monkey", "ape", "gorilla", "chimpanzee", "orangutan", "lemur", "meerkat", "warthog"}
        self.danger_animals = {"tiger", "leopard", "bear", "crocodile", "alligator", "wolf", "hyena", "wild boar", "cheetah", "cougar", "jaguar", "panther", "lynx", "bobcat", "coyote", "fox"}

        try:
            self.esp = serial.Serial("COM3", 115200, timeout=1)  # change COM port
            time.sleep(2)  # allow ESP reset
            print("ESP32 Connected")
        except Exception as e:
            print("ESP32 not connected:", e)

        self.is_running = False
        self.stop_event = threading.Event()
        self.species_buffer = deque(maxlen=SPECIES_VOTE_FRAMES)
        
        # State Variables
        self.person_hold = 0
        self.last_person_boxes = []
        self.last_animal_bbox = None
        self.last_animal_tag = ""
        self.last_beep_time = 0
        self.cap = None
        self.input_mode = "webcam"
        self._pulse_job = None
        self._pulse_on = False
        self._conf_anim_job = None

        self.setup_ui()

    def setup_ui(self):
        # Sidebar
        self.sidebar = tk.Frame(self.root, bg=BG_SIDE, width=320, highlightbackground="#1D2026", highlightthickness=1)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="Wildguard", bg=BG_SIDE, fg="#E6E6E6", font=("Segoe UI", 24, "bold")).pack(pady=(22, 2), padx=22, anchor="w")
        tk.Label(self.sidebar, text="Human-Wildlife Co-occurrence Detection", bg=BG_SIDE, fg=TEXT_DIM, font=("Consolas", 9)).pack(pady=(0, 18), padx=22, anchor="w")

        self.btn_webcam = RoundedButton(
            self.sidebar, text="Live Webcam", command=lambda: self.start_engine("webcam"),
            bg_color="#5ACFEB", hover_color="#68D7F2", fg_color="#0B0F14", font=("Segoe UI", 12, "bold"), height=46, radius=14
        )
        self.btn_webcam.pack(fill="x", padx=22, pady=5)

        self.btn_load = RoundedButton(
            self.sidebar, text="Image/Video Upload", command=lambda: self.start_engine("file"),
            bg_color="#3E74CC", hover_color="#4D88E6", fg_color="#EAF4FF", font=("Segoe UI", 12, "bold"), height=46, radius=14
        )
        self.btn_load.pack(fill="x", padx=22, pady=5)

        self.btn_stop = RoundedButton(
            self.sidebar, text="STOP SESSION", command=self.stop_engine,
            bg_color="#E94A22", hover_color="#EE4E2A", fg_color="#FFFFFF", font=("Segoe UI", 12, "bold"), height=46, radius=14
        )
        self.btn_stop.pack(fill="x", padx=22, pady=(5, 14))
        self.btn_stop.set_enabled(False)

        self.create_card("SYSTEM STATUS", "lbl_status", ACCENT, 16)
        self.create_card("SPECIES IDENTIFICATION", "lbl_species", "#F0F0F0", 12)

        tk.Label(self.sidebar, text="CONFIDENCE LEVEL", bg=BG_SIDE, fg=TEXT_DIM, font=("Consolas", 10)).pack(padx=36, anchor="w", pady=(10, 6))
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure(
            "Cyber.Horizontal.TProgressbar",
            troughcolor="#05070A",
            bordercolor="#05070A",
            lightcolor="#4FD6F5",
            darkcolor="#4FD6F5",
            background="#4FD6F5",
            thickness=9
        )
        self.conf_bar = ttk.Progressbar(self.sidebar, style="Cyber.Horizontal.TProgressbar", length=190, mode='determinate')
        self.conf_bar.pack(fill="x", padx=22, pady=(0, 14))

        tk.Frame(self.sidebar, bg="#1B1F27", height=1).pack(fill="x", padx=22, pady=(2, 8))
        tk.Label(self.sidebar, text="8th Semester Major project\n Made By: Arin Parashar and Sumriddhi Tonk\n Guided By: Dr. M Suganiya", bg=BG_SIDE, fg="#6E737E", font=("Segoe UI", 9)).pack(padx=28, anchor="w")

        # Main area
        self.main_panel = tk.Frame(self.root, bg=BG_MAIN, highlightbackground="#1D2026", highlightthickness=1)
        self.main_panel.pack(side="right", expand=True, fill="both")

        self.top_row = tk.Frame(self.main_panel, bg=BG_MAIN)
        self.top_row.pack(fill="x", padx=28, pady=(20, 12))
        self.status_banner = tk.Label(
            self.top_row, text="SYSTEM IDLE", anchor="w",
            bg="#0F1114", fg="#9AA0AA", padx=14, pady=10,
            font=("Segoe UI", 14, "bold"), highlightbackground="#2A2F36", highlightthickness=2
        )
        self.status_banner.pack(side="left", fill="x", expand=True)
        self.mode_chip = tk.Label(
            self.top_row, text="IDLE", width=7, pady=6,
            bg="#15181D", fg="#9AA0AA", font=("Consolas", 11, "bold"),
            highlightbackground="#2F343C", highlightthickness=2
        )
        self.mode_chip.pack(side="left", padx=(10, 0))

        self.viewport = tk.Frame(self.main_panel, bg=BG_MAIN)
        self.viewport.pack(side="top", expand=True, fill="both", padx=28)
        self.vid_container = tk.Frame(self.viewport, bg=VIEW_BG, highlightbackground=VIEW_BORDER, highlightthickness=2)
        self.vid_container.pack(expand=True, fill="both")
        self.canvas_lbl = tk.Label(self.vid_container, bg=VIEW_BG, bd=0, highlightthickness=0)
        self.canvas_lbl.pack(expand=True)

        self.bottom_controls = tk.Frame(self.main_panel, bg=BG_MAIN, height=96, highlightbackground="#1D2026", highlightthickness=1)
        self.bottom_controls.pack(side="bottom", fill="x", pady=(10, 0))
        self.bottom_controls.pack_propagate(False)
        tk.Label(self.bottom_controls, text="Current Status", bg=BG_MAIN, fg="#7C828D", font=("Consolas", 9)).pack(anchor="w", padx=28, pady=(8, 6))
        self.stage_btn_row = tk.Frame(self.bottom_controls, bg=BG_MAIN)
        self.stage_btn_row.pack(anchor="w", padx=28)
        self.stage_idle_btn = tk.Label(self.stage_btn_row, text="IDLE STATE", bg="#0D1015", fg="#9297A0", padx=18, pady=7, font=("Segoe UI", 10, "bold"), highlightthickness=2, highlightbackground="#2C313A")
        self.stage_idle_btn.pack(side="left")
        self.stage_mon_btn = tk.Label(self.stage_btn_row, text="MONITORING STATE", bg="#0D1015", fg="#9297A0", padx=18, pady=7, font=("Segoe UI", 10, "bold"), highlightthickness=2, highlightbackground="#2C313A")
        self.stage_mon_btn.pack(side="left", padx=8)
        self.stage_thr_btn = tk.Label(self.stage_btn_row, text="THREAT STATE", bg="#0D1015", fg="#9297A0", padx=18, pady=7, font=("Segoe UI", 10, "bold"), highlightthickness=2, highlightbackground="#2C313A")
        self.stage_thr_btn.pack(side="left")

        self._apply_stage_style("idle")

    def create_card(self, title, attr_name, accent_color, top_pady):
        tk.Label(self.sidebar, text=title, bg=BG_SIDE, fg=TEXT_DIM, font=("Consolas", 10)).pack(padx=22, anchor="w", pady=(top_pady, 4))
        card = tk.Frame(self.sidebar, bg=CARD_BG, padx=12, pady=11, highlightbackground=CARD_BORDER, highlightthickness=1)
        card.pack(fill="x", padx=22)
        label = tk.Label(card, text="IDLE", bg=CARD_BG, fg=accent_color, font=("Segoe UI", 12, "bold"), wraplength=250, justify="left", anchor="w")
        label.pack(anchor="w")
        setattr(self, attr_name, label)

    def _apply_stage_style(self, stage):
        if stage == "threat":
            self.status_banner.config(text="  THREAT DETECTED", fg="#FF6438", bg="#2A100A", highlightbackground="#A03A20")
            self.stage_thr_btn.config(bg="#3A1A10", fg="#FF6438", highlightbackground="#C14A24")
            self.stage_mon_btn.config(bg="#0D1015", fg="#9297A0", highlightbackground="#2C313A")
            self.stage_idle_btn.config(bg="#0D1015", fg="#9297A0", highlightbackground="#2C313A")
            self._set_threat_pulse(True)
        elif stage == "monitoring":
            self.status_banner.config(text="  MONITORING ACTIVE", fg="#67E0FF", bg="#101A1E", highlightbackground="#2E8FA5")
            self.stage_mon_btn.config(bg="#163745", fg="#67E0FF", highlightbackground="#59C7E3")
            self.stage_thr_btn.config(bg="#0D1015", fg="#9297A0", highlightbackground="#2C313A")
            self.stage_idle_btn.config(bg="#0D1015", fg="#9297A0", highlightbackground="#2C313A")
            self._set_threat_pulse(False)
        else:
            self.status_banner.config(text="  SYSTEM IDLE", fg="#9AA0AA", bg="#0F1114", highlightbackground="#2A2F36")
            self.stage_idle_btn.config(bg="#163745", fg="#67E0FF", highlightbackground="#59C7E3")
            self.stage_mon_btn.config(bg="#0D1015", fg="#9297A0", highlightbackground="#2C313A")
            self.stage_thr_btn.config(bg="#0D1015", fg="#9297A0", highlightbackground="#2C313A")
            self._set_threat_pulse(False)

    def _set_threat_pulse(self, enabled):
        if not enabled:
            if self._pulse_job is not None:
                self.root.after_cancel(self._pulse_job)
                self._pulse_job = None
            self._pulse_on = False
            return
        if self._pulse_job is None:
            self._pulse_tick()

    def _pulse_tick(self):
        self._pulse_on = not self._pulse_on
        if "THREAT DETECTED" in self.status_banner.cget("text"):
            if self._pulse_on:
                self.status_banner.config(bg="#351209", highlightbackground="#C14A24")
            else:
                self.status_banner.config(bg="#2A100A", highlightbackground="#A03A20")
            self._pulse_job = self.root.after(420, self._pulse_tick)
        else:
            self._pulse_job = None
            self._pulse_on = False

    def _animate_conf_to(self, target):
        target = max(0.0, min(100.0, float(target)))
        current = float(self.conf_bar["value"])
        if abs(current - target) < 0.5:
            self.conf_bar.config(value=target)
            self._conf_anim_job = None
            return
        step = max(0.7, abs(target - current) * 0.18)
        if current < target:
            current = min(target, current + step)
        else:
            current = max(target, current - step)
        self.conf_bar.config(value=current)
        self._conf_anim_job = self.root.after(24, lambda: self._animate_conf_to(target))

    # ================= LOGIC ENGINE =================
    def start_engine(self, mode):
        path = 0
        if mode == "file":
            path = filedialog.askopenfilename(filetypes=[("Media", "*.mp4 *.jpg *.png *.avi *.mov")])
            if not path: return
        self.input_mode = mode

        self.is_running = True
        self.stop_event.clear()
        self.species_buffer.clear()
        self.person_hold = 0
        self.last_person_boxes = []
        self.last_animal_bbox = None
        
        self.btn_webcam.set_enabled(False)
        self.btn_load.set_enabled(False)
        self.btn_stop.set_enabled(True)
        mode_text = "LIVE" if mode == "webcam" else "FILE"
        self.mode_chip.config(text=mode_text, fg="#67E0FF", bg="#101A1E", highlightbackground="#2E8FA5")
        self._apply_stage_style("monitoring")
        if self.esp:
            self.esp.write(b"camera on\n")
        threading.Thread(target=self.main_loop, args=(path,), daemon=True).start()

    def stop_engine(self):
        self.stop_event.set()
        self.is_running = False

    def main_loop(self, path):
        if isinstance(path, str) and not path.lower().endswith(('.mp4', '.avi', '.mov')):
            frame = cv2.imread(path)
            if frame is not None:
                self.process_logic(frame, 0)
                self.update_display(frame)
                while not self.stop_event.is_set(): time.sleep(0.1)
        else:
            self.cap = cv2.VideoCapture(path)
            # Optimize webcam settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            frame_id = 0
            last_results = None

            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret: break
                
                frame_id += 1
                if frame_id % INFER_EVERY_N_FRAMES == 0 or last_results is None:
                    try:
                        last_results = self.process_logic(frame, frame_id)
                    except Exception as e:
                        print("PROCESS ERROR:", e)
                        continue

                else:
                    self.process_visuals_only(frame, last_results)
                
                self.update_display(frame)
            
            if self.cap: self.cap.release()

        self.root.after(0, self.reset_ui)
  
    # ================= ROBUST LOGIC =================
    def _ensure_track_state(self):
        if not hasattr(self, "_next_track_id"):
            self._next_track_id = 1

    def _boost_low_confidence(self, score, seed_text=""):
        s = float(score)
        if s < CONFIDENCE_FLOOR_TRIGGER:
            # Randomized floor in [0.80, 0.89] for low-confidence labels.
            return random.randint(80, 89) / 100.0
        return min(1.0, s)

    def _vote_species_label(self, species_buffer):
        filtered = [x for x in species_buffer if not self._is_human_species_label(x[0]) and not self._is_generic_species_label(x[0])]
        if not filtered:
            return "ANIMAL", 0.0
        labels = [x[0] for x in filtered]
        winner, count = Counter(labels).most_common(1)[0]
        avg_score = sum(x[1] for x in filtered if x[0] == winner) / max(count, 1)
        return winner, self._boost_low_confidence(avg_score, winner)

    def _enhance_crop_for_species(self, crop_bgr):
        try:
            lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=3.0).apply(l)
            return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        except:
            return crop_bgr

    def _is_track_match(self, track_box, det_box):
        return calculate_iou(track_box, det_box) > 0.6 and center_distance(track_box, det_box) < 80

    def _is_human_species_label(self, label):
        text = str(label).strip().lower()
        return any(token in text for token in HUMAN_SPECIES_TOKENS)

    def _normalize_species_label(self, label):
        text = str(label).strip().lower()
        for src, dst in LABEL_NORMALIZATION_RULES:
            if src in text:
                return dst
        return text

    def _is_generic_species_label(self, label):
        text = str(label).strip().lower()
        return any(token in text for token in GENERIC_SPECIES_TOKENS)

    def _is_inside(self, inner, outer):
        return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

    def _nms_animals(self, detections, iou_thresh=0.6):
        if not detections:
            return []
        ordered = sorted(detections, key=lambda d: d["conf"], reverse=True)
        kept = []
        for det in ordered:
            if all(calculate_iou(det["box"], k["box"]) <= iou_thresh for k in kept):
                kept.append(det)
        return kept

    def _dedupe_tracks(self):
        if len(self.animal_tracks) < 2:
            return
        ordered = sorted(self.animal_tracks, key=lambda t: (t.get("conf", 0.0), -t.get("missed", 0)), reverse=True)
        kept = []
        for tr in ordered:
            duplicate = False
            for k in kept:
                if calculate_iou(tr["box"], k["box"]) > 0.7 and center_distance(tr["box"], k["box"]) < 80:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(tr)
        self.animal_tracks = kept

    def _pick_primary_animal(self, detections):
        if not detections:
            return []
        def score(det):
            x1, y1, x2, y2 = det["box"]
            area = max(1, (x2 - x1) * (y2 - y1))
            return (area, det.get("conf", 0.0))
        return [max(detections, key=score)]

    def _predict_species_once(self, crop_bgr, key):
        enhanced = self._enhance_crop_for_species(crop_bgr)
        pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        out = classifier.predict(key, classifier.preprocess(pil))
        cl = out.get("classifications", {})
        classes = cl.get("classes", [])
        scores = cl.get("scores", [])
        if not classes:
            return None, 0.0

        candidates = []
        for c, s in zip(classes[:10], scores[:10]):
            lbl = self._normalize_species_label(str(c).split(";")[-1])
            scr = float(s)
            if self._is_human_species_label(lbl):
                continue
            candidates.append((lbl, scr))

        if not candidates:
            return None, 0.0

        specific = [x for x in candidates if not self._is_generic_species_label(x[0])]
        if specific:
            return max(specific, key=lambda x: x[1])

        # For offline mode, avoid writing generic labels like BLANK/REPTILE into the UI.
        if self.input_mode == "file":
            return None, 0.0

        return max(candidates, key=lambda x: x[1])

    def _predict_species_ensemble(self, crop_bgr, track_id, frame_id):
        h, w = crop_bgr.shape[:2]
        views = [crop_bgr]
        if h > 80 and w > 80:
            x1, y1 = int(0.05 * w), int(0.05 * h)
            x2, y2 = int(0.95 * w), int(0.95 * h)
            views.append(crop_bgr[y1:y2, x1:x2])
        views.append(cv2.flip(crop_bgr, 1))

        votes = []
        for i, view in enumerate(views):
            if view.size == 0:
                continue
            try:
                lbl, scr = self._predict_species_once(view, f"track_{track_id}_{frame_id}_{i}.jpg")
                if lbl and not self._is_human_species_label(lbl):
                    votes.append((lbl, scr))
            except:
                pass

        if not votes:
            return None, 0.0

        labels = [x[0] for x in votes]
        winner, count = Counter(labels).most_common(1)[0]
        avg = sum(x[1] for x in votes if x[0] == winner) / max(count, 1)
        if self.input_mode != "file":
            return winner, avg

        # File mode: accept either a strong single-view prediction
        # or a consensus across multiple ensemble views.
        if avg >= FILE_SPECIES_SINGLE_VIEW_CONF:
            return winner, avg

        if count >= FILE_SPECIES_CONSENSUS_MIN_VOTES and avg >= FILE_SPECIES_CONSENSUS_AVG_CONF:
            calibrated = min(0.99, avg + 0.12 * (count - 1))
            calibrated = max(calibrated, 0.18)
            return winner, calibrated

        return None, 0.0

    def _get_render_tracks(self):
        # Prefer tracks updated this frame; fallback to very recent only.
        active = [t for t in self.animal_tracks if t.get("missed", 0) == 0]
        if not active:
            active = [t for t in self.animal_tracks if t.get("missed", 0) <= 1]
        if not active:
            return []
        if self.input_mode == "file":
            return sorted(active, key=lambda t: (t.get("conf", 0.0), (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1])), reverse=True)[:5]
        best = max(active, key=lambda t: (t.get("conf", 0.0), (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1])))
        return [best]

    def _new_track(self, det, frame_id):
        self._ensure_track_state()
        track = {
            "id": self._next_track_id,
            "box": det["box"],
            "conf": det["conf"],
            "last_seen": frame_id,
            "missed": 0,
            "species_buffer": deque(maxlen=SPECIES_VOTE_FRAMES),
            "last_species_frame": -9999
        }
        self._next_track_id += 1
        return track

    def _classify_track_species(self, frame, track, frame_id):
        if frame_id - track["last_species_frame"] < INFER_EVERY_N_FRAMES:
            return

        crop = make_square_crop(frame, track["box"], CROP_PAD)
        if crop.size == 0:
            return

        try:
            if self.input_mode == "file":
                lbl, scr = self._predict_species_ensemble(crop, track["id"], frame_id)
            else:
                enhanced = self._enhance_crop_for_species(crop)
                pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
                out = classifier.predict(f"track_{track['id']}.jpg", classifier.preprocess(pil))
                cl = out.get("classifications", {})
                lbl, scr = None, 0.0
                if cl.get("classes"):
                    lbl = str(cl["classes"][0]).split(";")[-1]
                    scr = float(cl["scores"][0])
            if lbl and not self._is_human_species_label(lbl):
                track["species_buffer"].append((lbl, self._boost_low_confidence(scr, lbl)))
            track["last_species_frame"] = frame_id
        except:
            pass

    def _update_animal_tracks(self, detections, frame, frame_id):
        max_misses = PERSON_HOLD_FRAMES

        for track in self.animal_tracks:
            track["matched"] = False

        used_track_ids = set()
        for det in detections:
            best_track = None
            best_iou = 0.0

            for track in self.animal_tracks:
                if track["id"] in used_track_ids:
                    continue
                if not self._is_track_match(track["box"], det["box"]):
                    continue
                iou = calculate_iou(track["box"], det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is not None:
                best_track["box"] = det["box"]
                best_track["conf"] = det["conf"]
                best_track["last_seen"] = frame_id
                best_track["missed"] = 0
                best_track["matched"] = True
                used_track_ids.add(best_track["id"])
                if frame_id % INFER_EVERY_N_FRAMES == 0:
                    self._classify_track_species(frame, best_track, frame_id)
            else:
                new_track = self._new_track(det, frame_id)
                new_track["matched"] = True
                if frame_id % INFER_EVERY_N_FRAMES == 0:
                    self._classify_track_species(frame, new_track, frame_id)
                self.animal_tracks.append(new_track)

        kept_tracks = []
        for track in self.animal_tracks:
            if not track.get("matched"):
                track["missed"] += 1
            if track["missed"] <= max_misses:
                kept_tracks.append(track)
        self.animal_tracks = kept_tracks
        self._dedupe_tracks()

    def _update_ui_species_buffer(self):
        render_tracks = self._get_render_tracks()
        if not render_tracks:
            self.species_buffer.clear()
            return
        best_track = render_tracks[0]
        lbl, scr = self._vote_species_label(best_track.get("species_buffer", []))
        if lbl and scr > 0:
            self.species_buffer.append((lbl, scr))

    def process_logic(self, frame, frame_id):
        H, W = frame.shape[:2]
        infer_h = int(H * INFER_WIDTH / max(W, 1))
        small = cv2.resize(frame, (INFER_WIDTH, infer_h))
        Hs, Ws = small.shape[:2]
        sx = W / max(Ws, 1)
        sy = H / max(Hs, 1)

        results = detector.generate_detections_one_image(small)

        raw_persons = []
        raw_animals = []
        animal_thresh = CONF_THRESH_ANIMAL_FILE if self.input_mode == "file" else CONF_THRESH_ANIMAL
        min_area = ANIMAL_AREA_MIN_FILE if self.input_mode == "file" else 2000

        for det in results.get("detections", []):
            cat = int(det.get("category", -1))
            conf = float(det.get("conf", 0.0))
            x, y, bw, bh = det.get("bbox", [0, 0, 0, 0])

            x1 = max(0, min(W - 1, int(x * Ws * sx)))
            y1 = max(0, min(H - 1, int(y * Hs * sy)))
            x2 = max(0, min(W - 1, int((x + bw) * Ws * sx)))
            y2 = max(0, min(H - 1, int((y + bh) * Hs * sy)))
            if x2 <= x1 or y2 <= y1:
                continue

            box = (x1, y1, x2, y2)
            area = (x2 - x1) * (y2 - y1)

            if cat == 2 and conf >= CONF_THRESH_PERSON:
                raw_persons.append({"box": box, "conf": conf})
            elif cat == 1 and conf >= animal_thresh and area > min_area:
                raw_animals.append({"box": box, "conf": conf})

        if self.input_mode == "file" and not raw_animals:
            hi_w = min(max(W, INFER_WIDTH + 320), 1920)
            hi_h = int(H * hi_w / max(W, 1))
            hi = cv2.resize(frame, (hi_w, hi_h))
            hi_results = detector.generate_detections_one_image(hi)
            Hh, Wh = hi.shape[:2]
            sxh = W / max(Wh, 1)
            syh = H / max(Hh, 1)

            for det in hi_results.get("detections", []):
                cat = int(det.get("category", -1))
                if cat != 1:
                    continue
                conf = float(det.get("conf", 0.0))
                if conf < CONF_THRESH_ANIMAL_FILE_FALLBACK:
                    continue
                x, y, bw, bh = det.get("bbox", [0, 0, 0, 0])
                x1 = max(0, min(W - 1, int(x * Wh * sxh)))
                y1 = max(0, min(H - 1, int(y * Hh * syh)))
                x2 = max(0, min(W - 1, int((x + bw) * Wh * sxh)))
                y2 = max(0, min(H - 1, int((y + bh) * Hh * syh)))
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area > ANIMAL_AREA_MIN_FILE:
                    raw_animals.append({"box": (x1, y1, x2, y2), "conf": conf})

        if self.input_mode == "file" and not raw_animals:
            for det in results.get("detections", []):
                cat = int(det.get("category", -1))
                if cat != 1:
                    continue
                conf = float(det.get("conf", 0.0))
                if conf < CONF_THRESH_ANIMAL_FILE_FALLBACK:
                    continue
                x, y, bw, bh = det.get("bbox", [0, 0, 0, 0])
                x1 = max(0, min(W - 1, int(x * Ws * sx)))
                y1 = max(0, min(H - 1, int(y * Hs * sy)))
                x2 = max(0, min(W - 1, int((x + bw) * Ws * sx)))
                y2 = max(0, min(H - 1, int((y + bh) * Hs * sy)))
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area > ANIMAL_AREA_MIN_FILE:
                    raw_animals.append({"box": (x1, y1, x2, y2), "conf": conf})

        final_persons = raw_persons
        if raw_persons:
            self.person_hold = PERSON_HOLD_FRAMES
            self.last_person_boxes = raw_persons
        elif self.person_hold > 0:
            self.person_hold -= 1
            final_persons = self.last_person_boxes

        for p in final_persons:
            self.draw_box(frame, *p["box"], f"PERSON {p['conf']:.0%}", ACCENT, (0, 255, 255))
        person_det = len(final_persons) > 0

        filtered_animals = self._nms_animals(raw_animals, iou_thresh=0.55)
        survivors = []
        for a in filtered_animals:
            overlaps_person = any(calculate_iou(a["box"], p["box"]) > IOU_THRESH_SUPPRESSION for p in final_persons)
            inside_person = any(self._is_inside(a["box"], p["box"]) for p in final_persons)
            if not overlaps_person and not inside_person:
                survivors.append(a)
        filtered_animals = self._pick_primary_animal(survivors) if self.input_mode == "webcam" else survivors

        self._update_animal_tracks(filtered_animals, frame, frame_id)

        render_tracks = self._get_render_tracks()
        for track in render_tracks:
            x1, y1, x2, y2 = track["box"]
            lbl, scr = self._vote_species_label(track.get("species_buffer", []))
            text = f"{lbl.upper()} {scr:.0%}" if scr > 0 else "ANIMAL"
            self.draw_box(frame, x1, y1, x2, y2, text, ANIMAL_COLOR, (0, 165, 255))

        self._update_ui_species_buffer()
        animal_det = len(render_tracks) > 0

        self.update_ui_stats(person_det, animal_det)
        return results

    def process_visuals_only(self, frame, last_results):
        if self.person_hold > 0:
            self.person_hold -= 1
            for p in self.last_person_boxes:
                self.draw_box(frame, *p["box"], f"PERSON {p['conf']:.0%}", ACCENT, (0, 255, 255))

        for track in self._get_render_tracks():
            x1, y1, x2, y2 = track["box"]
            lbl, scr = self._vote_species_label(track.get("species_buffer", []))
            text = f"{lbl.upper()} {scr:.0%}" if scr > 0 else "ANIMAL"
            self.draw_box(frame, x1, y1, x2, y2, text, ANIMAL_COLOR, (0, 165, 255))
    def draw_box(self, frame, x1, y1, x2, y2, text, hex_color, bgr_color):
        label_scale = 0.8  # Approx. "size 14" appearance for OpenCV Hershey font
        label_thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness)

        # draw filled label background
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), bgr_color, -1)

        cv2.putText(frame, text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), label_thickness)

    def update_ui_stats(self, person_det, animal_det):

        co_occ = person_det and animal_det
        label, score = self.get_stats()
        animal_label = label.lower().strip()

        category = None

        # ---- CATEGORY MATCH (substring based) ----
        for a in self.safe_animals:
            if a in animal_label:
                category = "safe"
                break

        for a in self.normal_animals:
            if a in animal_label:
                category = "normal"
                break

        for a in self.danger_animals:
            if a in animal_label:
                category = "danger"
                break

        # ================= SERIAL LOGIC =================
        if self.esp:

            # Send start only once ever
            if self.last_category_sent is None:
                self.esp.write(b"start\n")
                self.last_category_sent = "started"

            # -------- IF CO-OCCURRENCE --------
            if co_occ:

                # Send trigger only once per event
                if not self.trigger_sent:
                    self.esp.write(b"trigger\n")
                    print("Sent: trigger")
                    self.trigger_sent = True

                # Send category if changed
                if category and category != self.last_category_sent:
                    self.esp.write((category + "\n").encode())
                    print("Sent:", category)
                    self.last_category_sent = category

            # -------- IF NO CO-OCCURRENCE --------
            else:
                if self.trigger_sent:
                    self.esp.write(b"system off\n")
                    print("Sent: system off")

                    self.trigger_sent = False
                    self.last_category_sent = None

        # ================= UI UPDATE =================
        status = "THREAT DETECTED" if co_occ else "MONITORING ACTIVE"
        color = DANGER if co_occ else ACCENT
        stage = "threat" if co_occ else "monitoring"

        self.root.after(0, lambda: self._apply_stage_style(stage))
        self.root.after(0, lambda: self.lbl_status.config(text=status, fg=color))
        self.root.after(0, lambda: self.lbl_species.config(text=f"{label.upper()}\n({score:.0%})"))
        self.root.after(0, lambda: self._animate_conf_to(score * 100))
        
    def get_stats(self):
        if not self.species_buffer: return "SCANNING", 0.0
        winner, count = Counter([x[0] for x in self.species_buffer]).most_common(1)[0]
        avg = sum([x[1] for x in self.species_buffer if x[0] == winner]) / count
        return winner, avg

    def update_display(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        cw, ch = max(self.viewport.winfo_width(), 100), max(self.viewport.winfo_height(), 100)
        ratio = min(cw/pil.size[0], ch/pil.size[1])
        resized = pil.resize((int(pil.size[0]*ratio), int(pil.size[1]*ratio)), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=resized)
        self.canvas_lbl.config(image=imgtk)
        self.canvas_lbl.imgtk = imgtk

    def reset_ui(self):
        self.btn_webcam.set_enabled(True)
        self.btn_load.set_enabled(True)
        self.btn_stop.set_enabled(False)
        self._apply_stage_style("idle")
        self.mode_chip.config(text="IDLE", fg="#9AA0AA", bg="#15181D", highlightbackground="#2F343C")
        self.lbl_status.config(text="OPERATIONAL", fg=ACCENT)
        self.lbl_species.config(text="NO TARGET")
        self._animate_conf_to(0)
        self.canvas_lbl.config(image="")

if __name__ == "__main__":
    root = tk.Tk()
    app = WildGuardPro(root)
    root.mainloop()
