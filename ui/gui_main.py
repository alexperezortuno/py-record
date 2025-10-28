import sys
import subprocess
import threading
import time
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel,
    QTextEdit, QFileDialog, QLineEdit, QHBoxLayout, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, QSettings, QThread, pyqtSignal


class RecorderThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, env_path, output_dir):
        super().__init__()
        self.env_path = env_path
        self.output_dir = output_dir

    def run(self):
        cmd = ["meeting", "record", "--env", self.env_path, "--output", self.output_dir]
        self.log_signal.emit(f"▶️ Running: {' '.join(cmd)}\n")
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                self.log_signal.emit(line.strip())
            process.wait()
            self.log_signal.emit("\n✅ Recording finished.\n")
            self.finished_signal.emit("done")
        except Exception as e:
            self.log_signal.emit(f"❌ Error: {e}")


class TranscriptionThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, env_path, audio_path, model):
        super().__init__()
        self.env_path = env_path
        self.audio_path = audio_path
        self.model = model

    def run(self):
        try:
            cmd = [
                "meeting", "transcript",
                "--env", self.env_path,
                "--audio", self.audio_path,
                "--model", self.model.lower()
            ]
            self.log_signal.emit(f"▶️ Running: {' '.join(cmd)}\n")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                self.log_signal.emit(result.stdout)
            if result.stderr:
                self.log_signal.emit(result.stderr)
            self.log_signal.emit("\n✅ Transcription complete.\n")
        except Exception as e:
            self.log_signal.emit(f"❌ Transcription failed: {e}")


class MeetingRecorder(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meeting Recorder 🎙️")
        self.setGeometry(300, 200, 600, 500)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        self.settings = QSettings("AlexPerezOrtuño", "MeetingRecorder")
        self.env_path = self.settings.value("env_path", "")
        self.output_dir = self.settings.value("output_dir", os.path.expanduser("~/meetings"))

        # UI components (same as before)
        self.env_input = QLineEdit(self.env_path)
        self.browse_env_button = QPushButton("Browse")
        self.output_input = QLineEdit(self.output_dir)
        self.browse_output_button = QPushButton("Browse")
        self.status_label = QLabel("Ready to record.")
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.record_button = QPushButton("⏺️ Start Recording")
        self.stop_button = QPushButton("⏹️ Stop Recording")
        self.transcribe_button = QPushButton("🧠 Transcribe & Summarize")
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Gemini", "OpenAI"])

        # Layouts
        env_layout = QHBoxLayout()
        env_layout.addWidget(self.env_input)
        env_layout.addWidget(self.browse_env_button)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(self.browse_output_button)
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("AI Model:"))
        model_layout.addWidget(self.model_selector)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select .env file:"))
        layout.addLayout(env_layout)
        layout.addWidget(QLabel("Select output folder:"))
        layout.addLayout(output_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.log_view)
        layout.addLayout(model_layout)
        layout.addWidget(self.record_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.transcribe_button)
        self.setLayout(layout)

        # State
        self.record_thread = None
        self.transcribe_thread = None
        self.last_record_path = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.start_time = None

        # Signals
        self.browse_env_button.clicked.connect(self.browse_env)
        self.browse_output_button.clicked.connect(self.browse_output)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.transcribe_button.clicked.connect(self.transcribe_and_summarize)

        self.stop_button.setEnabled(False)
        self.transcribe_button.setEnabled(True)

    def browse_env(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select .env", os.getcwd(), "Env Files (*.env)")
        if file_path:
            self.env_path = file_path
            self.env_input.setText(file_path)
            self.settings.setValue("env_path", file_path)
            self.log_view.append(f"✅ Loaded .env: {file_path}")

    def browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Folder", os.getcwd())
        if dir_path:
            self.output_dir = dir_path
            self.output_input.setText(dir_path)
            self.settings.setValue("output_dir", dir_path)
            self.log_view.append(f"📁 Output folder: {dir_path}")

    def start_recording(self):
        if not os.path.exists(self.env_path):
            QMessageBox.critical(self, "Error", "Please select a valid .env file.")
            return

        self.status_label.setText("Recording...")
        self.log_view.append("🎙️ Starting recording...\n")
        self.start_time = time.time()
        self.timer.start(1000)

        self.record_thread = RecorderThread(self.env_path, self.output_dir)
        self.record_thread.log_signal.connect(self.log_view.append)
        self.record_thread.finished_signal.connect(self.recording_finished)
        self.record_thread.start()

        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_recording(self):
        self.log_view.append("\n🛑 Stop requested.\n")
        if self.record_thread and self.record_thread.isRunning():
            self.record_thread.terminate()  # safely stop thread
        self.status_label.setText("Recording stopped.")
        self.stop_button.setEnabled(False)
        self.transcribe_button.setEnabled(True)
        self.timer.stop()

    def recording_finished(self, _):
        self.log_view.append("✅ Recording thread ended.\n")
        self.status_label.setText("Recording finished.")
        self.transcribe_button.setEnabled(True)

    def update_timer(self):
        elapsed = int(time.time() - self.start_time)
        self.status_label.setText(f"Recording... {elapsed}s")

    def transcribe_and_summarize(self):
        try:
            model = self.model_selector.currentText()
            self.log_view.append(f"\n🧠 Transcribing with {model}...\n")
            files = [f for f in os.listdir(self.output_dir) if f.endswith(".mp3")]
            if not files:
                QMessageBox.warning(self, "No file", "No MP3 found.")
                return

            latest = max(files, key=lambda f: os.path.getmtime(os.path.join(self.output_dir, f)))
            audio_path = os.path.join(self.output_dir, latest)
            self.transcribe_thread = TranscriptionThread(self.env_path, audio_path, model)
            self.transcribe_thread.log_signal.connect(self.log_view.append)
            self.transcribe_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Transcription failed: {e}")
            self.transcribe_thread = None


if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "xcb"  # helps on Wayland
    app = QApplication(sys.argv)
    gui = MeetingRecorder()
    gui.show()
    sys.exit(app.exec())
