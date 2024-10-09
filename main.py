import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL import Image
import cv2
import numpy as np
import csv
import datetime
from detect import process_image, update_csv_file

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        cap = cv2.VideoCapture(0)  

        while True:
            ret, frame = cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)
            QThread.msleep(30)

        cap.release()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.start()

        self.recent_plates = []
        self.load_recent_plates()

    def initUI(self):
        self.setWindowTitle("License Plate Detection")
        self.setGeometry(100, 100, 1600, 750)

        main_layout = QVBoxLayout(self)

        camera_detect_layout = QHBoxLayout()

        self.camera_label = QLabel(self)
        self.camera_label.setMinimumSize(640, 480)
        camera_detect_layout.addWidget(self.camera_label)

        self.detect_label = QLabel(self)
        self.detect_label.setMinimumSize(640, 480)
        camera_detect_layout.addWidget(self.detect_label)

        main_layout.addLayout(camera_detect_layout)

        self.capture_button = QPushButton("Chụp ảnh", self)
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setMinimumWidth(100)
        self.capture_button.setStyleSheet("font-size: 16px; padding: 10px;")
        main_layout.addWidget(self.capture_button, alignment=Qt.AlignCenter)

        self.status_label = QLabel(self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 18px;")
        main_layout.addWidget(self.status_label)

        self.search_layout = QHBoxLayout()
        self.search_line_edit = QLineEdit(self)
        self.search_line_edit.setPlaceholderText("Nhập biển số cần tìm...")
        self.search_line_edit.setMaximumWidth(200)
        self.search_line_edit.textChanged.connect(self.on_search_text_changed)
        self.search_layout.addWidget(self.search_line_edit)

        self.search_button = QPushButton("Tìm kiếm", self)
        self.search_button.setMaximumWidth(100)
        self.search_button.setMinimumWidth(80)
        self.search_button.clicked.connect(self.search_plate)
        self.search_layout.addWidget(self.search_button)
        main_layout.addLayout(self.search_layout)

        self.plates_table = QTableWidget(self)
        self.plates_table.setColumnCount(3)
        self.plates_table.setHorizontalHeaderLabels(["Thời gian", "Biển số", "Trạng thái"])
        self.plates_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.plates_table.setMaximumHeight(150)
        main_layout.addWidget(self.plates_table)

        self.setLayout(main_layout)

    def capture_image(self):
        frame = self.last_frame.copy()
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        processed_image, recognized_texts, found_plate, status = process_image(image_pil)

        if found_plate:
            if recognized_texts:
                recognized_text = ''.join(recognized_texts).replace(" ", "")  # Loại bỏ khoảng trắng
                update_csv_file([recognized_text], status)
                self.status_label.setText(f"Xe biển số {recognized_text} đã đi {status}")
                current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                self.add_to_recent_plates(current_time, recognized_text, status)
            else:
                print("Không nhận diện được ký tự trên biển số.")

            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            h, w, ch = processed_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(processed_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_qt_format)
            self.detect_label.setPixmap(pixmap)
        else:
            print("Không tìm thấy biển số xe trong hình ảnh.")
            self.status_label.setText("Không tìm thấy biển số xe trong hình ảnh.")

    def update_image(self, frame):
        self.last_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)
        self.camera_label.setPixmap(pixmap)

    def load_recent_plates(self):
        self.clear_table()
        with open('number_plate.csv', mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)[-5:]
        
        for row in reversed(rows):  # Duyệt qua các hàng theo thứ tự ngược lại
            self.add_to_recent_plates(row[0], row[1], row[2])

    def add_to_recent_plates(self, time, plate_number, status):
        if self.plates_table.rowCount() >= 5:
            self.plates_table.removeRow(self.plates_table.rowCount() - 1)  # Xóa hàng cuối cùng nếu vượt quá giới hạn 5 hàng

        self.plates_table.insertRow(0)  # Chèn hàng mới vào đầu bảng
        self.plates_table.setItem(0, 0, QTableWidgetItem(time))
        self.plates_table.setItem(0, 1, QTableWidgetItem(plate_number))
        self.plates_table.setItem(0, 2, QTableWidgetItem(status))

    def search_plate(self):
        search_text = self.search_line_edit.text().strip().upper()
        if not search_text:
            self.load_recent_plates()
            return

        found = False
        with open('number_plate.csv', mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 3 and row[1].strip() == search_text:
                    self.clear_table()
                    self.add_to_recent_plates(row[0], row[1], row[2])
                    found = True
                    break

        if not found:
            QMessageBox.warning(self, "Không tìm thấy", f"Không tìm thấy biển số xe '{search_text}' trong danh sách.")

    def clear_table(self):
        while self.plates_table.rowCount() > 0:
            self.plates_table.removeRow(0)

    def on_search_text_changed(self):
        if not self.search_line_edit.text().strip():
            self.load_recent_plates()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
