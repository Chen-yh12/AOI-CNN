import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import warnings

# -------------------------- Reproducibility Settings --------------------------
warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# -------------------------- Path Settings --------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs("captures", exist_ok=True)

# -------------------------- AOI Image Preprocessing (Based on Paper 3.3) --------------------------
class AOIPreprocessor:
    def __init__(self):
        self.clip_limit = 2.0
        self.tile_grid_size = (8, 8)
        self.kernel_size = (3, 3)
        self.sigmaX = 0

    def grayscale_conversion(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def gaussian_blur(self, image):
        return cv2.GaussianBlur(image, self.kernel_size, self.sigmaX)

    def clahe_enhancement(self, image):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return clahe.apply(image)

    def edge_enhancement(self, image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobelx = cv2.convertScaleAbs(sobelx)
        abs_sobely = cv2.convertScaleAbs(sobely)
        return cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

    def illumination_correction(self, image):
        background = cv2.GaussianBlur(image, (15, 15), 0)
        return cv2.subtract(background, image)

    def process(self, image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        gray = self.grayscale_conversion(image_rgb)
        blur = self.gaussian_blur(gray)
        enhance = self.clahe_enhancement(blur)
        edge = self.edge_enhancement(enhance)
        corrected = self.illumination_correction(edge)
        
        result_rgb = cv2.cvtColor(corrected, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(result_rgb)

# -------------------------- CNN Model (Based on Paper 3.4: 5 Convolutional Blocks) --------------------------
class TrayDamageDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(TrayDamageDetector, self).__init__()
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -------------------------- Inference Engine --------------------------
class PalletDetector:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = TrayDamageDetector().to(self.device)
        self.model.eval()
        self.aoi_preprocessor = AOIPreprocessor()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    @torch.no_grad()
    def predict(self, img_path):
        processed_img = self.aoi_preprocessor.process(img_path)
        tensor = self.transform(processed_img).unsqueeze(0).to(self.device)
        outputs = self.model(tensor)
        _, predicted = torch.max(outputs.data, 1)
        return "Damaged Pallet" if predicted.item() == 1 else "Normal Pallet"

# -------------------------- GUI (All English) --------------------------
class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Pallet Defect Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg="#F0F2F5")
        self.detector = PalletDetector()
        self.img_tk = None
        self.setup_ui()

    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#2A4A7C", pady=12)
        title_frame.pack(fill=tk.X)
        tk.Label(title_frame, text="Intelligent Pallet Defect Detection System",
                font=("Arial", 20, "bold"), fg="white", bg="#2A4A7C").pack()

        # Main frame
        main_frame = tk.Frame(self.root, bg="#F0F2F5")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # Left control panel
        left_frame = tk.Frame(main_frame, bg="white", width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        left_frame.pack_propagate(0)
        tk.Label(left_frame, text="Detection Control", font=("Arial", 16, "bold"), bg="white").pack(pady=20)

        btn_style = {
            "font": ("Arial", 13), "bg": "#3D6CB9", "fg": "white",
            "width": 20, "height": 2, "bd": 0
        }
        tk.Button(left_frame, text="Upload Image", command=self.detect_upload, **btn_style).pack(pady=10)
        tk.Button(left_frame, text="Camera Capture", command=self.detect_camera, **btn_style).pack(pady=10)

        # Right display area
        right_frame = tk.Frame(main_frame, bg="#F0F2F5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Log
        log_frame = tk.LabelFrame(right_frame, text="Detection Results", font=("Arial", 12))
        log_frame.pack(fill=tk.X, pady=5)
        self.log = scrolledtext.ScrolledText(log_frame, height=10, font=("Arial", 11))
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Image preview
        img_frame = tk.LabelFrame(right_frame, text="Image Preview", font=("Arial", 12))
        img_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.preview = tk.Label(img_frame, bg="white")
        self.preview.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.log_msg("System started successfully → Ready for pallet inspection")

    def log_msg(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def show_full_image(self, path):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(1000 / w, 600 / h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(img)
        self.preview.config(image=self.img_tk)

    def detect_upload(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if not path:
            return

        self.show_full_image(path)
        result = self.detector.predict(path)

        self.log_msg("\n========================================")
        self.log_msg(f"Image inspection completed")
        self.log_msg(f"Result: {result}")
        self.log_msg("========================================\n")

    def detect_camera(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            messagebox.showwarning("Warning", "Camera failed to open!")
            cap.release()
            return
        path = "captures/camera_capture.jpg"
        cv2.imwrite(path, frame)
        cap.release()

        self.show_full_image(path)
        result = self.detector.predict(path)

        self.log_msg("\n========================================")
        self.log_msg(f"Camera capture inspection completed")
        self.log_msg(f"Result: {result}")
        self.log_msg("========================================\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainGUI(root)
    root.mainloop()
