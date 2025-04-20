import cv2
import numpy as np
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import logging
import os
import csv
import pandas as pd
from ultralytics import YOLO
from collections import deque

class WorkerActivityMonitor:
    def __init__(self, root, video_source=0):
        self.root = root
        self.root.title("Worker Activity Monitor")
        self.root.geometry("1400x800")
        self.root.state('zoomed')  # Start maximized
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.LOG_FILE = "activity_log.csv"
        self.REPORT_DIR = "reports"

        
        self.IDLE_THRESHOLD = 5  # seconds
        self.MOTION_SENSITIVITY = 1000  # Pixels for motion detection
        self.ALERT_THRESHOLD = 30  # Seconds for idle alert
        self.running = False
        self.monitoring_active = False
        self.video_source = video_source
        self.record_video = False
        self.video_writer = None
        
        # State
        self.people = []
        self.person_counter = 0
        self.alerts = deque(maxlen=10)
        self.alert_sound_enabled = True
        
        # Initialize YOLOv8
        try:
            self.model = YOLO('yolov8n.pt')
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 model: {e}")
            messagebox.showerror("Error", "Failed to load YOLOv8 model!")
            self.root.destroy()
            return
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            self.logger.error("Failed to open video source")
            messagebox.showerror("Error", "Could not open video source!")
            self.root.destroy()
            return
        
        # Create report directory
        os.makedirs(self.REPORT_DIR, exist_ok=True)
        
        # Initialize UI
        self.create_ui()
    
    def create_ui(self):
        """Create professional UI with control panel buttons always visible at the top"""
        # Configure styles
        self.configure_styles()
        
        # Main container with paned windows
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        
        # Left frame (70% width) with control panel at the top
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=7)
        
        # Right pane (30% width)
        right_pane = ttk.PanedWindow(main_pane, orient=tk.VERTICAL)
        main_pane.add(right_pane, weight=3)
        
        # Create control panel at the top of left frame
        self.create_control_panel(left_frame)
        
        # Vertical paned window for video and log panels
        left_pane = ttk.PanedWindow(left_frame, orient=tk.VERTICAL)
        left_pane.pack(fill=tk.BOTH, expand=True)
        
        # Create video and log panels
        self.create_video_panel(left_pane)
        self.create_log_panel(left_pane)
        
        # Create status and alerts panels
        self.create_status_panel(right_pane)
        self.create_alerts_panel(right_pane)
        
        # Initialize video preview
        self.init_video_preview()
    
    def configure_styles(self):
        """Configure professional UI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Color scheme
        self.bg_color = '#1e2a38'
        self.card_bg = '#2c3e50'
        self.text_color = '#ecf0f1'
        self.primary_color = '#3498db'
        self.success_color = '#27ae60'
        self.danger_color = '#e74c3c'
        self.warning_color = '#f39c12'
        self.purple_color = '#9b59b6'
        
        # Configure styles
        style.configure('.', background=self.bg_color, foreground=self.text_color)
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabel', background=self.bg_color, foreground=self.text_color, font=('Helvetica', 10))
        style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Card.TFrame', background=self.card_bg, relief=tk.RAISED, borderwidth=1)
        
        # Button styles
        style.configure('Primary.TButton', background=self.primary_color, foreground='white')
        style.configure('Success.TButton', background=self.success_color, foreground='white')
        style.configure('Danger.TButton', background=self.danger_color, foreground='white')
        style.configure('Warning.TButton', background=self.warning_color, foreground='white')
        style.configure('Purple.TButton', background=self.purple_color, foreground='white')
        
        style.map('TButton',
                background=[('active', '#2980b9'), ('disabled', '#7f8c8d')])
        
        # Status label styles
        style.configure('Working.TLabel', foreground=self.success_color)
        style.configure('Idle.TLabel', foreground=self.danger_color)
    
    def create_video_panel(self, parent):
        """Create video display panel"""
        video_card = ttk.Frame(parent, style='Card.TFrame', padding=10)
        parent.add(video_card, weight=2)
        
        # Header with stats
        header = ttk.Frame(video_card)
        header.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header, text="LIVE VIDEO FEED", style='Title.TLabel').pack(side=tk.LEFT)
        
        stats = ttk.Frame(header)
        stats.pack(side=tk.RIGHT)
        
        self.worker_count_var = tk.StringVar(value="👥 Workers: 0")
        ttk.Label(stats, textvariable=self.worker_count_var, style='Title.TLabel').pack(side=tk.LEFT, padx=10)
        
        self.active_count_var = tk.StringVar(value="✅ Active: 0")
        ttk.Label(stats, textvariable=self.active_count_var, style='Title.TLabel').pack(side=tk.LEFT)
        
        # Video display
        self.video_label = ttk.Label(video_card)
        self.video_label.pack(fill=tk.BOTH, expand=True)
    
    def create_control_panel(self, parent):
        """Create control button panel with buttons always visible"""
        control_card = ttk.Frame(parent, style='Card.TFrame', padding=10)
        control_card.pack(fill=tk.X, side=tk.TOP)
        
        # Set fixed height for control panel
        control_card.pack_propagate(False)
        control_card.configure(height=100)  # Sufficient height for buttons
        
        # Button container with grid layout
        button_frame = ttk.Frame(control_card)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Configure grid to center buttons
        button_frame.columnconfigure((0, 1, 2, 3), weight=1)
        button_frame.rowconfigure(0, weight=1)
        
        # Buttons with fixed width for compact layout
        self.start_btn = ttk.Button(
            button_frame, text="▶ START MONITORING", 
            command=self.start_monitoring, style='Success.TButton', width=18
        )
        self.start_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.stop_btn = ttk.Button(
            button_frame, text="⏹ STOP MONITORING", 
            command=self.stop_monitoring, style='Danger.TButton', state=tk.DISABLED, width=18
        )
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Button(
            button_frame, text="⚙ SETTINGS", 
            command=self.show_settings, style='Primary.TButton', width=18
        ).grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        ttk.Button(
            button_frame, text="📊 EXPORT REPORT", 
            command=self.export_report, style='Purple.TButton', width=18
        ).grid(row=0, column=3, padx=5, pady=5, sticky="ew")
    
    def create_log_panel(self, parent):
        """Create activity log panel with proper scrolling"""
        log_card = ttk.Frame(parent, style='Card.TFrame', padding=10)
        parent.add(log_card, weight=1)
        
        ttk.Label(log_card, text="ACTIVITY LOG", style='Title.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        # Create frame with scrollbar
        log_frame = ttk.Frame(log_card)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(
            log_frame, wrap=tk.WORD, 
            bg=self.card_bg, fg=self.text_color, 
            font=('Consolas', 10), state=tk.DISABLED
        )
        
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure text tags for colors
        self.log_text.tag_config('working', foreground=self.success_color)
        self.log_text.tag_config('idle', foreground=self.danger_color)
        self.log_text.tag_config('alert', foreground=self.warning_color)
        self.log_text.tag_config('system', foreground=self.primary_color)
    
    def create_status_panel(self, parent):
        """Create worker status panel with scrolling"""
        status_card = ttk.Frame(parent, style='Card.TFrame', padding=10)
        parent.add(status_card, weight=1)
        
        ttk.Label(status_card, text="WORKER STATUS", style='Title.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        # Create canvas with scrollbar
        canvas_frame = ttk.Frame(status_card)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_canvas = tk.Canvas(
            canvas_frame, bg=self.card_bg, 
            highlightthickness=0, bd=0
        )
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.status_canvas.yview)
        self.status_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Inner frame for status items
        self.status_frame = ttk.Frame(self.status_canvas)
        self.status_canvas.create_window((0, 0), window=self.status_frame, anchor="nw")
        
        # Configure scrolling
        self.status_frame.bind(
            "<Configure>",
            lambda e: self.status_canvas.configure(
                scrollregion=self.status_canvas.bbox("all")
            )
        )
    
    def create_alerts_panel(self, parent):
        """Create alerts panel with scrolling"""
        alert_card = ttk.Frame(parent, style='Card.TFrame', padding=10)
        parent.add(alert_card, weight=1)
        
        ttk.Label(alert_card, text="ALERTS", style='Title.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        # Create text widget with scrollbar
        alert_frame = ttk.Frame(alert_card)
        alert_frame.pack(fill=tk.BOTH, expand=True)
        
        self.alert_text = tk.Text(
            alert_frame, wrap=tk.WORD, 
            bg=self.card_bg, fg=self.warning_color,
            font=('Consolas', 10), state=tk.DISABLED
        )
        
        scrollbar = ttk.Scrollbar(alert_frame, orient=tk.VERTICAL, command=self.alert_text.yview)
        self.alert_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.alert_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def init_video_preview(self):
        """Show video feed without processing"""
        def update_preview():
            if not self.monitoring_active:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (900, 600))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
                else:
                    self.logger.warning("Failed to read frame during preview")
            self.root.after(100, update_preview)
        update_preview()
    
    def start_monitoring(self):
        """Start monitoring process"""
        self.monitoring_active = True
        self.running = True
        self.people = []
        self.person_counter = 0
        self.alerts.clear()
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Clear logs
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.alert_text.config(state=tk.NORMAL)
        self.alert_text.delete(1.0, tk.END)
        self.alert_text.config(state=tk.DISABLED)
        
        # Clear status frame
        for widget in self.status_frame.winfo_children():
            widget.destroy()
        
        # Initialize video writer if recording
        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monitoring_{timestamp}.avi"
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (800, 500))
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_worker, daemon=True)
        self.monitor_thread.start()
        
        self.log_activity("SYSTEM", "Monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring process"""
        self.running = False
        self.monitoring_active = False
        
        # Release video writer if recording
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Update UI
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        self.log_activity("SYSTEM", "Monitoring stopped")
        self.init_video_preview()
    
    def monitor_worker(self):
        """Main monitoring function"""
        prev_frame = None
        roi_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        
        while self.running and self.monitoring_active:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("Failed to read frame")
                if isinstance(self.video_source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.1)
                continue
            
            # Preprocess frame
            frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Person detection
            results = self.model(frame, conf=0.5, classes=[0])
            persons = []
            for result in results:
                for box in result.boxes:
                    x, y, w, h = box.xywh[0].cpu().numpy().astype(int)
                    x, y = x - w//2, y - h//2
                    x, y, w, h = max(0, x-10), max(0, y-10), min(320-x, w+20), min(240-y, h+20)
                    if w > 10 and h > 10:
                        persons.append((x, y, w, h))
            
            # Update worker count
            self.root.after(0, self.update_worker_count, len(persons))
            
            # Track people
            new_people = []
            for i, (x, y, w, h) in enumerate(persons):
                roi = (x, y, w, h)
                matched = False
                
                for person in self.people:
                    px, py, pw, ph = person[0]
                    if abs(x - px) < 50 and abs(y - py) < 50:
                        new_people.append((roi, person[1], person[2], person[3], person[4]))
                        matched = True
                        break
                
                if not matched:
                    self.person_counter += 1
                    person_id = f"Worker {self.person_counter}"
                    motion_history = deque(maxlen=10)
                    new_people.append((roi, "IDLE", time.time(), person_id, motion_history))
                    self.logger.info(f"New person detected: {person_id}")
                    self.root.after(0, self.add_status_label, person_id)
            
            self.people = new_people
            
            # Process motion
            display_frame = cv2.resize(frame, (800, 500))
            scale_x, scale_y = 800/320, 500/240
            active_count = 0
            
            for i, (roi, state, last_motion_time, person_id, motion_history) in enumerate(self.people):
                x, y, w, h = roi
                roi_gray = gray[y:y+h, x:x+w]
                
                motion_detected = False
                if prev_frame is not None:
                    prev_roi = prev_frame[y:y+h, x:x+w]
                    if prev_roi.shape == roi_gray.shape:
                        frame_diff = cv2.absdiff(prev_roi, roi_gray)
                        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                        motion_count = cv2.countNonZero(thresh)
                        motion_detected = motion_count > self.MOTION_SENSITIVITY
                        motion_history.append(motion_detected)
                
                # Update state
                current_time = time.time()
                if motion_detected:
                    last_motion_time = current_time
                    if state != "WORKING":
                        state = "WORKING"
                        self.root.after(0, self.update_status_label, person_id, state)
                        self.log_activity(person_id, "Working activity detected")
                elif current_time - last_motion_time > self.IDLE_THRESHOLD:
                    if state != "IDLE":
                        state = "IDLE"
                        self.root.after(0, self.update_status_label, person_id, state)
                        self.log_activity(person_id, "Idle state detected")
                
                # Check for alerts
                if current_time - last_motion_time > self.ALERT_THRESHOLD and state == "IDLE":
                    alert_msg = f"ALERT: {person_id} has been idle for {int(current_time - last_motion_time)} seconds!"
                    if alert_msg not in self.alerts:
                        self.alerts.append(alert_msg)
                        self.root.after(0, self.add_alert, alert_msg)
                        self.log_activity(person_id, alert_msg, alert=True)
                
                if state == "WORKING":
                    active_count += 1
                
                self.people[i] = (roi, state, last_motion_time, person_id, motion_history)
                
                # Draw on frame
                color = roi_colors[i % len(roi_colors)]
                dx, dy, dw, dh = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)
                cv2.rectangle(display_frame, (dx, dy), (dx+dw, dy+dh), color, 2)
                
                # Motion indicator
                motion_ratio = sum(motion_history)/len(motion_history) if motion_history else 0
                motion_color = (0, 255, 0) if motion_ratio > 0.5 else (0, 0, 255) if motion_ratio > 0 else (0, 255, 255)
                cv2.rectangle(display_frame, (dx, dy-5), (dx + int(dw*motion_ratio), dy-2), motion_color, -1)
                
                label = f"{person_id}: {state}"
                cv2.putText(display_frame, label, (dx, dy-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, ), 2, cv2.LINE_AA)
            
            # Update active count
            self.root.after(0, self.update_active_count, active_count)
            
            # Write frame if recording
            if self.record_video and self.video_writer is not None:
                self.video_writer.write(display_frame)
            
            # Update display
            self.update_video_feed(display_frame)
            
            prev_frame = gray.copy()
            time.sleep(0.1)
        
        if not self.monitoring_active:
            self.cap.release()
    
    def add_status_label(self, person_id):
        """Add status label for a new person"""
        frame = ttk.Frame(self.status_frame)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=person_id, width=15, anchor=tk.W).pack(side=tk.LEFT)
        
        status_var = tk.StringVar(value="IDLE")
        status_label = ttk.Label(frame, textvariable=status_var, style='Idle.TLabel')
        status_label.pack(side=tk.LEFT, padx=10)
        
        if not hasattr(self, 'status_widgets'):
            self.status_widgets = {}
        self.status_widgets[person_id] = (frame, status_var, status_label)
        
        self.status_canvas.configure(scrollregion=self.status_canvas.bbox("all"))
    
    def update_status_label(self, person_id, state):
        """Update status label for a person"""
        if hasattr(self, 'status_widgets') and person_id in self.status_widgets:
            _, status_var, status_label = self.status_widgets[person_id]
            status_var.set(state)
            style = 'Working.TLabel' if state == "WORKING" else 'Idle.TLabel'
            status_label.configure(style=style)
    
    def update_worker_count(self, count):
        """Update the worker count display"""
        self.worker_count_var.set(f"👥 Workers: {count}")
    
    def update_active_count(self, count):
        """Update the active worker count display"""
        self.active_count_var.set(f"✅ Active: {count}")
    
    def update_video_feed(self, frame):
        """Update video feed display"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def add_alert(self, alert_msg):
        """Add alert to alert panel"""
        self.alert_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.alert_text.insert(tk.END, f"[{timestamp}] {alert_msg}\n")
        self.alert_text.see(tk.END)
        self.alert_text.config(state=tk.DISABLED)
        
        # Visual alert
        self.alert_text.config(bg='#4a2c0a')
        self.root.after(200, lambda: self.alert_text.config(bg=self.card_bg))
        
        # Sound alert
        if self.alert_sound_enabled:
            try:
                import winsound
                winsound.Beep(1000, 500)
            except:
                pass
    
    def log_activity(self, source, message, alert=False):
        """Log activity with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp},{source},{message}\n"
        
        # Update UI log
        self.log_text.config(state=tk.NORMAL)
        display_entry = f"[{timestamp}] {source}: {message}\n"
        if alert:
            tag = 'alert'
        elif "Working" in message:
            tag = 'working'
        elif "Idle" in message:
            tag = 'idle'
        else:
            tag = 'system'
        self.log_text.insert(tk.END, display_entry, tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Save to file
        try:
            with open(self.LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, source, message])
        except Exception as e:
            self.logger.error(f"Failed to write to log file: {e}")
    
    def export_report(self):
        """Export activity report"""
        try:
            df = pd.read_csv(self.LOG_FILE, names=['Timestamp', 'Source', 'Activity'])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.REPORT_DIR, f"activity_report_{timestamp}.xlsx")
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='All Activity', index=False)
                summary = df.groupby(['Source', 'Activity']).size().unstack(fill_value=0)
                summary.to_excel(writer, sheet_name='Summary')
                
                for worker in df['Source'].unique():
                    if worker != "SYSTEM":
                        worker_df = df[df['Source'] == worker]
                        worker_df.to_excel(writer, sheet_name=worker[:20], index=False)
            
            messagebox.showinfo("Export Successful", f"Report saved as:\n{filename}")
            self.log_activity("SYSTEM", f"Exported report: {filename}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Error exporting report:\n{str(e)}")
            self.logger.error(f"Failed to export report: {e}")
    
    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x400")
        
        # Settings notebook
        notebook = ttk.Notebook(settings_window)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Detection settings
        detect_frame = ttk.Frame(notebook)
        notebook.add(detect_frame, text="Detection")
        
        ttk.Label(detect_frame, text="Motion Sensitivity:").pack(pady=10)
        sensitivity_slider = ttk.Scale(detect_frame, from_=100, to=2000, value=self.MOTION_SENSITIVITY)
        sensitivity_slider.pack()
        
        ttk.Label(detect_frame, text="Idle Threshold (seconds):").pack(pady=10)
        threshold_slider = ttk.Scale(detect_frame, from_=1, to=30, value=self.IDLE_THRESHOLD)
        threshold_slider.pack()
        
        ttk.Label(detect_frame, text="Alert Threshold (seconds):").pack(pady=10)
        alert_slider = ttk.Scale(detect_frame, from_=5, to=120, value=self.ALERT_THRESHOLD)
        alert_slider.pack()
        
        # Video settings
        video_frame = ttk.Frame(notebook)
        notebook.add(video_frame, text="Video")
        
        self.record_var = tk.BooleanVar(value=self.record_video)
        ttk.Checkbutton(video_frame, text="Record Monitoring Session", variable=self.record_var).pack(pady=10)
        
        ttk.Label(video_frame, text="Video Source:").pack(pady=5)
        
        source_frame = ttk.Frame(video_frame)
        source_frame.pack()
        
        def browse_video():
            filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
            if filename:
                self.video_source = filename
                self.cap = cv2.VideoCapture(self.video_source)
        
        ttk.Button(source_frame, text="Webcam", command=lambda: setattr(self, 'video_source', 0)).pack(side=tk.LEFT, padx=5)
        ttk.Button(source_frame, text="Browse...", command=browse_video).pack(side=tk.LEFT, padx=5)
        
        # Alert settings
        alert_frame = ttk.Frame(notebook)
        notebook.add(alert_frame, text="Alerts")
        
        self.sound_var = tk.BooleanVar(value=self.alert_sound_enabled)
        ttk.Checkbutton(alert_frame, text="Enable Alert Sounds", variable=self.sound_var).pack(pady=10)
        
        def save_settings():
            self.MOTION_SENSITIVITY = int(sensitivity_slider.get())
            self.IDLE_THRESHOLD = int(threshold_slider.get())
            self.ALERT_THRESHOLD = int(alert_slider.get())
            self.record_video = self.record_var.get()
            self.alert_sound_enabled = self.sound_var.get()
            
            self.log_activity("SYSTEM", 
                f"Settings updated - Sensitivity: {self.MOTION_SENSITIVITY}, "
                f"Threshold: {self.IDLE_THRESHOLD}s, "
                f"Alert: {self.ALERT_THRESHOLD}s, "
                f"Record: {self.record_video}, "
                f"Sound: {self.alert_sound_enabled}"
            )
            settings_window.destroy()
        
        ttk.Button(settings_window, text="Save Settings", command=save_settings).pack(pady=20)
    
    def on_close(self):
        """Clean up before closing"""
        if self.monitoring_active:
            self.stop_monitoring()
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
        
        self.root.destroy()

if __name__ == "__main__":
    video_source = 0  # Use webcam for real-time
    root = tk.Tk()
    app = WorkerActivityMonitor(root, video_source)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()