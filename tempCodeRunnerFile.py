import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import pyttsx3

# ---------------------- GLOBAL IMAGE STORAGE ----------------------
GLOBAL_IMAGES = []

# ---------------------- Project Modules ----------------------
import show_attendance
import takeImage
import trainImage
import automaticAttedance
# ---------------------- Paths ----------------------
HAAR_PATH = "haarcascade_frontalface_default.xml"
TRAIN_LABEL_PATH = "./TrainingImageLabel/Trainner.yml"
TRAIN_IMAGE_DIR = "./TrainingImage"
studentdetail_path = "./StudentDetails/studentdetails.csv"
attendance_path = "./Attendance"

os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(attendance_path, exist_ok=True)

# ---------------------- TTS ----------------------
def text_to_speech(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# ---------------------- Theme ----------------------
BG = "#0f1113"
CARD_BG = "#15171a"
ACCENT = "#f2d300"
TEXT = "#e6e6e6"
SUBTEXT = "#9aa0a6"
BTN_BG = "#101214"
BTN_ACTIVE = "#1b1f22"
TITLE_FONT = ("Segoe UI", 26, "bold")

# ---------------------- Tk Root MUST be here FIRST ----------------------
root = tk.Tk()
root.title("CLASS VISION — Face Recognizer")
root.configure(bg=BG)
root.geometry("1280x720")

# Center window
root.update_idletasks()
sw = root.winfo_screenwidth()
sh = root.winfo_screenheight()
root.geometry(f"1280x720+{(sw//2)-640}+{(sh//2)-360}")

# ---------------------- Safe Image Loader ----------------------
def load_image_safe(path, size=None):
    if not os.path.exists(path):
        print("[WARNING] Missing image:", path)
        return None
    try:
        img = Image.open(path)
        if size:
            img = img.resize(size, Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        GLOBAL_IMAGES.append(tk_img)
        return tk_img
    except Exception as e:
        print("[ERROR] Image load failed:", e)
        return None

# ---------------------- HEADER ----------------------
header = tk.Frame(root, bg=BG)
header.pack(fill="x", padx=28, pady=(18,6))

def build_header():
    global logo_img

    logo_img = load_image_safe("UI_Image/0001.png", (52,48))
    if logo_img:
        logo_lbl = tk.Label(header, image=logo_img, bg=BG)
    else:
        logo_lbl = tk.Label(header, text="CV", fg=ACCENT, bg=BG, font=("Segoe UI", 30, "bold"))

    logo_lbl.pack(side="left", padx=(6,12))

    tk.Label(header, text="CLASS VISION", bg=BG, fg=ACCENT,
             font=TITLE_FONT).pack(side="left")

    tk.Label(header, text="Face recognition attendance made simple",
             bg=BG, fg=SUBTEXT, font=("Segoe UI", 13)).pack(side="left", padx=10)

root.after(50, build_header)

# ---------------------- Separator ----------------------
tk.Frame(root, bg="#0b0b0b", height=1).pack(fill="x", padx=28, pady=(10,20))

# ---------------------- Cards ----------------------
content = tk.Frame(root, bg=BG)
content.pack(expand=True, fill="both", padx=28)

cards = tk.Frame(content, bg=BG)
cards.place(relx=0.5, rely=0.12, anchor="n")

row = tk.Frame(cards, bg=BG)
row.pack()

# Card Data
img_paths = ["UI_Image/register.png", "UI_Image/attendance.png", "UI_Image/verifyy.png"]
titles = ["Register Student", "Take Attendance", "View Attendance"]
descs = [
    "Capture student images and prepare dataset.",
    "Start live face-recognition attendance.",
    "View or export attendance records."
]
btns = ["Register", "Start", "View"]

# Card Actions
def open_register():
    top = tk.Toplevel(root)
    top.title("Register Student")
    top.geometry("750x450")
    top.configure(bg=BG)
    top.resizable(False, False)

    tk.Label(top, text="Register Student Face", bg=BG, fg=ACCENT,
             font=("Segoe UI", 22, "bold")).pack(pady=10)

    form = tk.Frame(top, bg=BG)
    form.pack(padx=40, pady=10, fill="both")

    # Enrollment
    tk.Label(form, text="Enrollment No", bg=BG, fg=TEXT,
             font=("Segoe UI", 12)).grid(row=0, column=0, sticky="w")
    ent_enroll = tk.Entry(form, bg="#111214", fg=ACCENT,
                          insertbackground=ACCENT, font=("Segoe UI", 12),
                          relief="flat", bd=0)
    ent_enroll.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    # Name
    tk.Label(form, text="Name", bg=BG, fg=TEXT,
             font=("Segoe UI", 12)).grid(row=1, column=0, sticky="w")
    ent_name = tk.Entry(form, bg="#111214", fg=ACCENT,
                        insertbackground=ACCENT, font=("Segoe UI", 12),
                        relief="flat", bd=0)
    ent_name.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

    form.columnconfigure(1, weight=1)

    # Notification label
    note_lbl = tk.Label(top, text="", bg=BG, fg=SUBTEXT, font=("Segoe UI", 11))
    note_lbl.pack(pady=(10, 5))

    # Take Image
    def take_img_action():
        enrollment = ent_enroll.get().strip()
        name = ent_name.get().strip()
        takeImage.TakeImage(enrollment, name, HAAR_PATH,
                            TRAIN_IMAGE_DIR, note_lbl, None, text_to_speech)
        ent_enroll.delete(0, "end")
        ent_name.delete(0, "end")

    # Train Model
    def train_img_action():
        trainImage.TrainImage(HAAR_PATH, TRAIN_IMAGE_DIR,
                              TRAIN_LABEL_PATH, note_lbl, text_to_speech)

    btn_row = tk.Frame(top, bg=BG)
    btn_row.pack(pady=10)

    tk.Button(btn_row, text="Take Image", command=take_img_action,
              bg=BTN_BG, fg=ACCENT, font=("Segoe UI", 12, "bold"),
              relief="flat", padx=18, pady=8, cursor="hand2").pack(side="left", padx=10)

    tk.Button(btn_row, text="Train Model", command=train_img_action,
              bg=BTN_BG, fg=ACCENT, font=("Segoe UI", 12, "bold"),
              relief="flat", padx=18, pady=8, cursor="hand2").pack(side="left", padx=10)


def start_attendance():
    automaticAttedance.subjectChoose(text_to_speech)

def view_attendance():
    show_attendance.subjectchoose(text_to_speech)

actions = [open_register, start_attendance, view_attendance]

# Create Cards
for i in range(3):
    card = tk.Frame(row, bg=CARD_BG, width=350, height=200)
    card.grid(row=0, column=i, padx=20)
    card.grid_propagate(False)

    img = load_image_safe(img_paths[i], (90, 90))
    if img:
        tk.Label(card, image=img, bg=CARD_BG).place(x=18, y=20)

    tk.Label(card, text=titles[i], bg=CARD_BG, fg=TEXT,
             font=("Segoe UI", 14, "bold")).place(x=140, y=18)

    tk.Label(card, text=descs[i], bg=CARD_BG, fg=SUBTEXT,
             wraplength=180, justify="left").place(x=140, y=55)

    b = tk.Button(card, text=btns[i], command=actions[i],
                  bg=BTN_BG, fg=ACCENT, bd=0, font=("Segoe UI", 11, "bold"))
    b.place(x=220, y=140, width=100, height=36)

# ---------------------- Footer ----------------------
footer = tk.Frame(root, bg=BG)
footer.pack(fill="x", padx=28, pady=18)

tk.Label(footer, text="CLASS VISION — v1.0", bg=BG, fg=SUBTEXT).pack(side="left")

def on_exit():
    root.quit()

tk.Button(footer, text="EXIT", bg=BTN_BG, fg=ACCENT,
          bd=0, command=on_exit).pack(side="right")

root.bind("<Escape>", lambda e: on_exit())

# ---------------------- Run ----------------------
root.mainloop()
