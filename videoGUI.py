import sys
import traceback
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QGridLayout, QLabel, QSlider, QPushButton, QHBoxLayout, QCheckBox, QDialog
from PyQt5.QtCore import QTimer, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import psycopg2
import cv2
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Define paths for the video files
video_path_right = "./Ahmed Bukhatir - Ya Adheeman (Lyrics) - With English Subtitles.mp4"
video_path_left = "./Be Happy - Motivational Nasheed - Othman Al Ibrahim.mp4"

class VideoPlayerWidget(QWidget):
    def __init__(self, video_path, target_size):
        super().__init__()

        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(self.video_path)
        if not self.video_capture.isOpened():
            raise ValueError(f"Error opening video file: {self.video_path}")

        self.target_size = target_size  # Desired size for displaying the video
        self.playing = True  # Flag to indicate video playback status
        self.mutex = QMutex()  # Mutex for thread safety

        self.video_label = QLabel(self)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.video_label)
        self.setLayout(self.layout)

    def set_frame(self, position):
        with QMutexLocker(self.mutex):  # Ensure exclusive access to video capture
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, position)

    def update_frame(self, frame):
        resized_frame = cv2.resize(frame, self.target_size)
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = resized_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(resized_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

    def reset_video(self):
        self.set_frame(0)  # Reset the video to the first frame
        self.playing = True  # Start playing the video

    def closeEvent(self, event):
        self.video_capture.release()  # Release the video capture
        super().closeEvent(event)  # Ensure the base class closeEvent is called

class VideoThread(QtCore.QThread):
    frame_signal = QtCore.pyqtSignal(object)
    position_signal = QtCore.pyqtSignal(int)

    def __init__(self, video_widget, sync_slider):
        super().__init__()
        self.video_widget = video_widget
        self.sync_slider = sync_slider
        self.running = True

    def run(self):
        while self.running:
            if self.video_widget.playing:
                with QMutexLocker(self.video_widget.mutex):  # Ensure exclusive access to video capture
                    ret, frame = self.video_widget.video_capture.read()
                if ret:
                    self.frame_signal.emit(frame)
                    current_frame = int(self.video_widget.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                    self.position_signal.emit(current_frame)
                    fps = self.video_widget.video_capture.get(cv2.CAP_PROP_FPS)
                    QtCore.QThread.msleep(int(1000 / fps))  # Adjust sleep duration for frame rate
                else:
                    self.video_widget.reset_video()  # Restart the video when it ends
            else:
                QtCore.QThread.msleep(100)

    def stop(self):
        self.running = False
        self.wait()

class MyMainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_sync_callback = None
        self.click_counter = 0

        layout = QGridLayout(self)

        self.figure1 = Figure(figsize=(4, 4))
        self.ax1 = self.figure1.add_subplot(111)
        self.canvas1 = FigureCanvas(self.figure1)
        layout.addWidget(self.canvas1, 0, 0)

        self.figure2 = Figure(figsize=(4, 4))
        self.ax2 = self.figure2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.figure2)
        layout.addWidget(self.canvas2, 0, 1)

        self.db_params = {
            "user": "postgres",
            "password": "0Sms@800008",
            "host": "localhost",
            "port": "5432",
            "database": "fft_db"
        }

        self.previous_train_no = None  # Variable to store the previous maximum train_no
        self.batch_size = 100000

        self.toolbar = NavigationToolbar(self.canvas1, self)
        layout.addWidget(self.toolbar, 1, 0, 1, 2)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.plot_data)
        self.timer.start(1000)  # Check for new data every second

        self.plot_data()  # Initial plot

        self.canvas1.mpl_connect('button_press_event', self.on_click)
        self.canvas2.mpl_connect('button_press_event', self.on_click)

        # Connect canvas events to sync limits
        self.canvas1.mpl_connect('draw_event', self.sync_axes)
        self.ax1.callbacks.connect('xlim_changed', self.sync_axes)
        self.ax1.callbacks.connect('ylim_changed', self.sync_axes)

    def on_click(self, event):
        if event.inaxes is not None:
            xdata = event.xdata
            if self.video_sync_callback is not None:
                self.click_counter += 1
                self.video_sync_callback((xdata, xdata), self.click_counter)

    def plot_data(self):
        try:
            # Connect to the PostgreSQL database
            with psycopg2.connect(**self.db_params) as mydb:
                with mydb.cursor() as mycursor:
                    # Execute the SQL query to fetch the maximum train_no
                    mycursor.execute(
                        'SELECT MAX("train_no") FROM "train_data_rms"'
                    )

                    max_train_no = mycursor.fetchone()[0]

                    # Check if the maximum train_no has changed
                    if max_train_no != self.previous_train_no:
                        # Execute the SQL query to fetch data for the new maximum train_no
                        mycursor.execute(
                            'SELECT rms_chan_0,rms_chan_1,rms_chan_2,rms_chan_3,rms_chan_4,rms_chan_5,rms_chan_6,rms_chan_7 FROM "train_data_rms" WHERE "train_no" = %s',
                            (max_train_no,)
                        )

                        result = mycursor.fetchall()

                        # Clear previous data points
                        self.ax1.clear()
                        self.ax2.clear()

                        # Check if there is any new data
                        if result:
                            # Convert fetched data to a numpy array
                            data_array = np.array(result)
                            # Generate x-axis values (assuming one data point per second)
                            serial_numbers = np.arange(1, len(data_array) + 1) / 1000

                            # Plot the first four sensors on the first subplot
                            self.ax1.plot(serial_numbers, data_array[:, 0], label="VHF(A)", linestyle="solid", lw=0.5)
                            self.ax1.plot(serial_numbers, data_array[:, 1], label="VHF(B)", linestyle="solid", lw=0.5)
                            self.ax1.plot(serial_numbers, data_array[:, 2], label="VLF(A)", linestyle="dotted", lw=0.5)
                            self.ax1.plot(serial_numbers, data_array[:, 3], label="VLF(B)", linestyle="dotted", lw=0.5)
                            self.ax1.legend()
                            self.ax1.set_xlim(0)
                            self.ax1.set_ylim(0, 11)  # Set y-axis to start from 0 to 11 for the first subplot
                            self.ax1.set_xlabel('Time in Sec')
                            self.ax1.set_ylabel('Voltage Range')

                            # Plot the next four sensors on the second subplot
                            self.ax2.plot(serial_numbers, data_array[:, 4], label="SHF(A)", linestyle="dotted", lw=0.5)
                            self.ax2.plot(serial_numbers, data_array[:, 5], label="SHF(B)", linestyle="dotted", lw=0.5)
                            self.ax2.plot(serial_numbers, data_array[:, 6], label="SLF(A)", linestyle="dotted", lw=0.5)
                            self.ax2.plot(serial_numbers, data_array[:, 7], label="SLF(B)", linestyle="dotted", lw=0.5)
                            self.ax2.legend()
                            self.ax2.set_xlim(0)
                            self.ax2.set_ylim(0, 11)  # Set y-axis to start from 0 to 11 for the second subplot
                            self.ax2.set_xlabel('Time in Sec')
                            self.ax2.set_ylabel('Voltage Range')

                            # Draw the updated plots
                            self.canvas1.draw()
                            self.canvas2.draw()

                        # Update the previous maximum train_no
                        self.previous_train_no = max_train_no
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            traceback.print_exc()  # Print the traceback for detailed error information

    def sync_axes(self, event=None):
        self.ax2.set_xlim(self.ax1.get_xlim())
        self.ax2.set_ylim(self.ax1.get_ylim())
        self.canvas2.draw()

class FFTDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FFT Graphs")
        self.setGeometry(0, 0, 800, 600)  # Set dialog size

        layout = QVBoxLayout(self)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.channel_select_layout = QHBoxLayout()

        self.select_all_button = QPushButton("Select All", self)
        self.select_all_button.clicked.connect(self.select_all_channels)
        layout.addWidget(self.select_all_button)

        self.clear_all_button = QPushButton("Clear All", self)
        self.clear_all_button.clicked.connect(self.clear_all_channels)
        layout.addWidget(self.clear_all_button)

        channel_names = ["VHF (A)", "VHF (B)", "VLF (A)", "VLF (B)", "SHF (A)", "SHF (B)", "SLF (A)", "SLF (B)"]

        self.channel_checkboxes = []
        for i in range(8):
            checkbox = QCheckBox(channel_names[i])
            self.channel_select_layout.addWidget(checkbox)
            self.channel_checkboxes.append(checkbox)

        layout.addLayout(self.channel_select_layout)

        self.generate_button = QPushButton("Generate Graph", self)
        self.generate_button.clicked.connect(self.generate_graph)
        layout.addWidget(self.generate_button)

        self.setLayout(layout)

    def select_all_channels(self):
        for checkbox in self.channel_checkboxes:
            checkbox.setChecked(True)

    def clear_all_channels(self):
        for checkbox in self.channel_checkboxes:
            checkbox.setChecked(False)

    def generate_graph(self):
        selected_channels = [i for i, checkbox in enumerate(self.channel_checkboxes) if checkbox.isChecked()]
        self.plot_fft(selected_channels)

    def plot_fft(self, selected_channels):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Fetch the latest data from the main widget's plot_data method
        try:
            with psycopg2.connect(**self.main_widget.db_params) as mydb:
                with mydb.cursor() as mycursor:
                    mycursor.execute(
                        'SELECT MAX("train_no") FROM "train_data_raw"'
                    )
                    max_train_no = mycursor.fetchone()[0]
                    mycursor.execute(
                        'SELECT ch0,ch1,ch2,ch3,ch4,ch5,ch6,ch7 FROM "train_data_raw" WHERE "train_no" = %s',
                        (max_train_no,)
                    )
                    result = mycursor.fetchall()
                    if result:
                        data_array = np.array(result)
                        for i in selected_channels:
                            y = data_array[:, i]
                            N = len(y)
                            T = 1.0 / 800.0  # Assuming a sample rate of 800 Hz for the example
                            yf = np.fft.fft(y)
                            xf = np.fft.fftfreq(N, T)[:N//2]
                            ax.plot(xf, 2.0/N * np.abs(yf[:N//2]), label=f'{self.channel_checkboxes[i].text()}')
        
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"Error fetching data for FFT: {e}")
            traceback.print_exc()
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 1200, 800)  # Adjust window size as needed

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        video_layout = QGridLayout()

        target_size = (600, 280)
        self.video_right = VideoPlayerWidget(video_path_right, target_size)
        video_layout.addWidget(self.video_right, 0, 0)

        self.video_left = VideoPlayerWidget(video_path_left, target_size)
        video_layout.addWidget(self.video_left, 0, 1)

        # Add the slider just below the videos
        self.slider = QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, int(cv2.VideoCapture(video_path_right).get(cv2.CAP_PROP_FRAME_COUNT)))
        self.slider.sliderMoved.connect(self.on_slider_moved)
        layout.addLayout(video_layout)
        layout.addWidget(self.slider)

        # Add Play and Pause buttons below the slider
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Play", self)
        self.start_button.clicked.connect(self.start_videos)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Pause", self)
        self.stop_button.clicked.connect(self.stop_videos)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        self.main_widget = MyMainWindow(self)
        layout.addWidget(self.main_widget)

        self.main_widget.video_sync_callback = self.sync_videos

        self.video_right_thread = VideoThread(self.video_right, self.slider)
        self.video_right_thread.frame_signal.connect(self.video_right.update_frame)
        self.video_right_thread.position_signal.connect(self.sync_slider_position)
        self.video_right_thread.start()

        self.video_left_thread = VideoThread(self.video_left, self.slider)
        self.video_left_thread.frame_signal.connect(self.video_left.update_frame)
        self.video_left_thread.position_signal.connect(self.sync_slider_position)
        self.video_left_thread.start()

        # Add "Show FFT" button to open a new window with FFT functionality
        self.fft_button = QPushButton("Show FFT", self)
        self.fft_button.clicked.connect(self.show_fft_window)
        layout.addWidget(self.fft_button)

    def sync_slider_position(self, position):
        self.slider.blockSignals(True)
        self.slider.setValue(position)
        self.slider.blockSignals(False)

    def start_videos(self):
        self.video_right.playing = True
        self.video_left.playing = True

    def stop_videos(self):
        self.video_right.playing = False
        self.video_left.playing = False

    def on_slider_moved(self, position):
        self.video_right.set_frame(position)
        self.video_left.set_frame(position)

    def sync_videos(self, x_limits, click_counter):
        frame_rate = self.video_right.video_capture.get(cv2.CAP_PROP_FPS)
        start_frame = int(x_limits[0] * frame_rate)
        end_frame = int(x_limits[1] * frame_rate)
        self.video_right.set_frame(start_frame)
        self.video_left.set_frame(start_frame)

        # Reset and play the videos as many times as the number of clicks
        for _ in range(click_counter):  
            self.video_right.reset_video()
            self.video_left.reset_video()

    def show_fft_window(self):
        self.fft_dialog = FFTDialog()
        self.fft_dialog.main_widget = self.main_widget  # Provide access to the main widget
        self.fft_dialog.show()

    def closeEvent(self, event):
        self.video_right_thread.stop()
        self.video_left_thread.stop()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
