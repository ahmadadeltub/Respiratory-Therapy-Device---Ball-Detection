import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from picamera2 import Picamera2

# =============================================================================
# Configuration Constants
# =============================================================================
# Colors and Fonts
BACKGROUND_COLOR = "#1a2a5a"  # dark blue
TEXT_COLOR = "white"
FONT_NAME = "Segoe UI"
RULER_FONT_SIZE = 12

# Canvas and Layout Settings
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 720
HEADER_HEIGHT = 90
CANVAS_WIDTH = WINDOW_WIDTH  # full window width
CANVAS_HEIGHT = WINDOW_HEIGHT - HEADER_HEIGHT

# Ruler settings (set margins to 0 so that the ruler fits exactly within the canvas)
RULER_X = 60              # ruler x-position
RULER_TOP_MARGIN = 50     # top margin for ruler (set to 0 to start at the top)
RULER_BOTTOM_MARGIN = 0   # bottom margin for ruler (set to 0 to end at the bottom)

# HSV Ranges for Color Detection
HSV_RANGES = {
    "Blue": {
        "lower": np.array([94, 80, 2]),
        "upper": np.array([126, 255, 255]),
        "draw_color": (255, 0, 0)
    },
    "Orange": {
        "lower": np.array([10, 100, 20]),
        "upper": np.array([25, 255, 255]),
        "draw_color": (0, 165, 255)
    },
    "Green": {
        # Adjusted green range for improved detection.
        "lower": np.array([35, 100, 20]),
        "upper": np.array([85, 255, 255]),
        "draw_color": (0, 100, 0)
    },
}

# Morphological operation kernel
KERNEL = np.ones((5, 5), np.uint8)

# =============================================================================
# Main Application Class
# =============================================================================
class RespiratoryTherapyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Respiratory Therapy Device")
        self.root.configure(bg=BACKGROUND_COLOR)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        # Dictionary to store ball positions
        self.ball_positions = {key: None for key in HSV_RANGES.keys()}
        self.last_frame_height = 480  # default; will update with captured frame

        # Initialize Camera
        self.init_camera()

        # Setup GUI Components
        self.setup_header()
        self.setup_main_panel()

        # Start frame update loop
        self.root.after(0, self.update_frame)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_camera(self):
        try:
            self.picam2 = Picamera2()
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.picam2.configure(preview_config)
            self.picam2.start()
        except Exception as e:
            print("Error initializing camera:", e)
            self.picam2 = None

    # -------------------------------
    # GUI Setup
    # -------------------------------
    def setup_header(self):
        """Set up the header with logos and title."""
        self.header_frame = tk.Frame(self.root, bg=BACKGROUND_COLOR, height=HEADER_HEIGHT)
        self.header_frame.pack(side="top", fill="x", pady=(0, 5))
        self.header_frame.pack_propagate(False)

        # Load logo images; use placeholder images if unavailable.
        try:
            logo_left = Image.open("qstss.png")
            logo_right = Image.open("moe.png")
        except Exception as e:
            print("Error loading logo images:", e)
            logo_left = Image.new("RGB", (60, 60), color="white")
            logo_right = Image.new("RGB", (60, 60), color="white")

        logo_left = logo_left.resize((80, 50))
        logo_right = logo_right.resize((80, 50))
        self.logo_left_img = ImageTk.PhotoImage(logo_left)
        self.logo_right_img = ImageTk.PhotoImage(logo_right)

        left_logo_label = tk.Label(self.header_frame, image=self.logo_left_img, bg=BACKGROUND_COLOR)
        left_logo_label.pack(side="left", padx=5)
        title_label = tk.Label(
            self.header_frame,
            text="Respiratory Therapy Device",
            font=(FONT_NAME, 18, "bold"),
            bg=BACKGROUND_COLOR,
            fg=TEXT_COLOR,
        )
        title_label.pack(side="left", padx=10)
        right_logo_label = tk.Label(self.header_frame, image=self.logo_right_img, bg=BACKGROUND_COLOR)
        right_logo_label.pack(side="right", padx=5)

    def setup_main_panel(self):
        """Create the main panel and canvas for ball indicators and ruler."""
        self.main_frame = tk.Frame(self.root, bg=BACKGROUND_COLOR, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
        self.main_frame.pack(side="top", fill="both", expand=True)
        self.main_frame.pack_propagate(False)

        self.canvas = tk.Canvas(self.main_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
                                bg=BACKGROUND_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Draw vertical ruler on the left spanning the entire canvas height
        self.draw_ruler()

        # Create ball indicator circles (initially placed at the bottom)
        default_y = CANVAS_HEIGHT
        radius = 20
        # Shifted to the left compared to the original positions
        self.blue_circle = self.canvas.create_oval(150 - radius, default_y - radius, 150 + radius, default_y + radius,
                                                    fill="white", outline="blue", width=3)
        self.orange_circle = self.canvas.create_oval(200 - radius, default_y - radius, 200 + radius, default_y + radius,
                                                      fill="white", outline="orange", width=3)
        self.Green_circle = self.canvas.create_oval(250 - radius, default_y - radius, 250 + radius, default_y + radius,
                                                         fill="white", outline="green", width=3)

        # Create percentage text items on the canvas
        self.blue_percent_text = self.canvas.create_text(200, 30, text="Blue: 0%", font=(FONT_NAME, 12), fill="white")
        self.orange_percent_text = self.canvas.create_text(300, 30, text="Orange: 0%", font=(FONT_NAME, 12), fill="white")
        self.Green_percent_text = self.canvas.create_text(400, 30, text="Green: 0%", font=(FONT_NAME, 12), fill="white")
        self.breathing_avg_text = self.canvas.create_text(300, 5, text="Breathing avg: 0%", font=(FONT_NAME, 15), fill="yellow")

    # -------------------------------
    # Image Processing and Ball Detection
    # -------------------------------
    @staticmethod
    def process_mask(mask):
        """
        Apply morphological operations (opening and closing) to reduce noise
        and improve detection accuracy.
        """
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
        return mask

    def detect_ball(self, mask, label, draw_color, frame):
        """
        Detect the largest contour in the mask, draw a circle and label on the frame,
        and update the ball_positions dictionary.
        """
        mask = self.process_mask(mask)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pos = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:  # Minimum size filter to avoid noise
                cv2.circle(frame, (int(x), int(y)), int(radius), draw_color, 2)
                cv2.putText(frame, label, (int(x - radius), int(y - radius)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
                pos = (int(x), int(y))
        self.ball_positions[label] = pos
        return frame

    # -------------------------------
    # Utility Methods for UI
    # -------------------------------
    def draw_ruler(self):
        """
        Draw a vertical white ruler along the left side of the canvas.
        The ruler spans the full height of the canvas and displays values
        from 100 at the top to 0 at the bottom.
        """
        effective_height = CANVAS_HEIGHT - RULER_TOP_MARGIN - RULER_BOTTOM_MARGIN
        ruler_top = RULER_TOP_MARGIN
        ruler_bottom = CANVAS_HEIGHT - RULER_BOTTOM_MARGIN

        # Draw the main ruler line.
        self.canvas.create_line(RULER_X, ruler_top, RULER_X, ruler_bottom, width=10, fill="white")

        # Draw tick marks and labels in steps of 10.
        for value in range(0, 101, 10):
            # Calculate y so that value 100 is at the top and 0 at the bottom.
            y = ruler_top + (1 - value / 100) * effective_height
            self.canvas.create_line(RULER_X, y, RULER_X + 50, y, width=4, fill="white")
            self.canvas.create_text(RULER_X - 25, y, text=str(value),
                                    font=(FONT_NAME, RULER_FONT_SIZE), fill="white")

    def get_canvas_y(self, ball_y):
        """
        Map the camera y-coordinate of a detected ball to a canvas y-coordinate.
        With this direct mapping:
          - ball_y = 0 (top of the frame) maps to 0 (top of the canvas)
          - ball_y = last_frame_height (bottom of the frame) maps to CANVAS_HEIGHT (bottom of the canvas)
        """
        normalized = ball_y / self.last_frame_height  # 0 at top, 1 at bottom
        normalized = max(0, min(normalized, 1))
        return normalized * CANVAS_HEIGHT

    def update_ball_indicators(self):
        """Update ball indicator positions and display percentages on the canvas."""
        def get_new_center(label):
            if self.ball_positions[label]:
                return self.get_canvas_y(self.ball_positions[label][1])
            else:
                return CANVAS_HEIGHT  # Default: bottom of canvas

        radius = 20
        blue_y = get_new_center("Blue")
        orange_y = get_new_center("Orange")
        Green_y = get_new_center("Green")

        self.canvas.coords(self.blue_circle, 200 - radius, blue_y - radius, 210 + radius, blue_y + radius)
        self.canvas.itemconfig(self.blue_circle, fill="blue" if self.ball_positions["Blue"] else "white")

        self.canvas.coords(self.orange_circle, 300 - radius, orange_y - radius, 310 + radius, orange_y + radius)
        self.canvas.itemconfig(self.orange_circle, fill="orange" if self.ball_positions["Orange"] else "white")

        self.canvas.coords(self.Green_circle, 400 - radius, Green_y - radius, 410 + radius, Green_y + radius)
        self.canvas.itemconfig(self.Green_circle, fill="Green" if self.ball_positions["Green"] else "white")

        # Calculate and display percentages.
        blue_percent = int(round((1 - (self.ball_positions["Blue"][1] / self.last_frame_height)) * 100)) if self.ball_positions["Blue"] else 0
        orange_percent = int(round((1 - (self.ball_positions["Orange"][1] / self.last_frame_height)) * 100)) if self.ball_positions["Orange"] else 0
        Green_percent = int(round((1 - (self.ball_positions["Green"][1] / self.last_frame_height)) * 100)) if self.ball_positions["Green"] else 0

        self.canvas.itemconfig(self.blue_percent_text, text=f"Blue: {blue_percent}%")
        self.canvas.itemconfig(self.orange_percent_text, text=f"Orange: {orange_percent}%")
        self.canvas.itemconfig(self.Green_percent_text, text=f"Green: {Green_percent}%")

        avg = int(round((blue_percent + orange_percent + Green_percent) / 3))
        self.canvas.itemconfig(self.breathing_avg_text, text=f"Breathing avg: {avg}%")

    # -------------------------------
    # Main Loop: Frame Capture and Processing
    # -------------------------------
    def update_frame(self):
        """Capture a frame, process it for ball detection, and update the UI accordingly."""
        if self.picam2 is None:
            print("Camera not initialized.")
            self.root.after(10, self.update_frame)
            return

        frame = self.picam2.capture_array()
        if frame is None:
            print("Error: Could not capture frame from camera.")
            self.root.after(10, self.update_frame)
            return

        self.last_frame_height = frame.shape[0]
        frame = cv2.flip(frame, 1)  # Mirror effect
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        for label, settings in HSV_RANGES.items():
            mask = cv2.inRange(hsv, settings["lower"], settings["upper"])
            frame = self.detect_ball(mask, label, settings["draw_color"], frame)

        self.update_ball_indicators()

        # Uncomment the following lines to display the processed frame for debugging:
        # cv2.imshow("Processed Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     self.on_closing()

        self.root.after(10, self.update_frame)

    def on_closing(self):
        """Clean up resources when closing the application."""
        if self.picam2 is not None:
            self.picam2.stop()
        self.root.destroy()


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = RespiratoryTherapyApp(root)
    root.mainloop()
