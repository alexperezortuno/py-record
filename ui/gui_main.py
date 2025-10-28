import sys
import subprocess
import threading
import time
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer


class MeetingRecorder(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meeting Recorder 🎙️")
        self.setGeometry(300, 200, 400, 300)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        # Widgets
        self.status_label = QLabel("Ready to record.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self.record_button = QPushButton("⏺️ Start Recording")
        self.stop_button = QPushButton("⏹️ Stop Recording")
        self.stop_button.setEnabled(False)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.log_view)
        layout.addWidget(self.record_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        # Events
        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)

        # Recording process
        self.record_process = None
        self.start_time = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)

    def start_recording(self):
        """Start the CLI recording process."""
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Recording... Press Stop to end.")
        self.start_time = time.time()
        self.timer.start(1000)

        # Run in background thread
        def run_record():
            cmd = ["meeting", "record", "--env", ".env"]
            self.log_view.append(f"Running: {' '.join(cmd)}\n")
            self.record_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in self.record_process.stdout:
                self.log_view.append(line.strip())
            self.record_process.wait()
            self.log_view.append("\n✅ Recording finished.\n")

        threading.Thread(target=run_record, daemon=True).start()

    def stop_recording(self):
        """Send stop signal to CLI."""
        if self.record_process:
            self.record_process.terminate()  # sends SIGTERM (Linux)
            self.log_view.append("\n🛑 Stop signal sent.\n")

        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Recording stopped.")
        self.timer.stop()

    def update_timer(self):
        elapsed = int(time.time() - self.start_time)
        self.status_label.setText(f"Recording... {elapsed}s")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MeetingRecorder()
    gui.show()
    sys.exit(app.exec())
