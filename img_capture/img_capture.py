import win32api, win32gui, win32ui, win32service, win32con
import os, os.path
from PIL import Image
from PIL import ImageGrab
import cv2 as cv
import numpy as np
import time
import sys
from PyQt5.QtWidgets import (QWidget, QPushButton, QFrame, QApplication)
from PyQt5.QtGui import QColor

from matplotlib import pyplot as plt

## find window title
# time.sleep(10)
# tempWindowName=win32gui.GetWindowText(win32gui.GetForegroundWindow())
# print(tempWindowName)


def find_window():
    appname = "Samsung Galaxy S7"
    hwnd = win32gui.FindWindow(None, appname)
    hwnd2 = win32gui.GetWindow(hwnd, 2)
    capture_window(hwnd2)

def capture_window(hwnd2):
    DIR = 'img/background'
    number = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    bbox1 = win32gui.GetWindowRect(hwnd2)
    img = ImageGrab.grab(bbox1)
    img_np = np.array(img)
    cv.imwrite('img/background'+'/back_{}.png'.format(number),img_np) #saved BGR

class Capture_manager(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
        
    def initUI(self):                       
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Capture manager')

        self.start_btn = QPushButton('Start',self)
        self.start_btn.move(45, 65)

        self.start_btn.clicked[bool].connect(self.start_capture)

        self.show()

    def start_capture(self, pressed):
        start_time = time.time()
        self.start_btn.setDisabled(True)
        self.start_btn.setText('wait')
        for i in range(10):
            find_window()
            time.sleep(0.5)
            if(i%10==0):
                self.start_btn.setText('wait...{}%'.format(i))
            if(i >=99):
                self.start_btn.setText('Start')
                self.start_btn.setEnabled(True)

def main():
    find_window()

if __name__=='__main__':
    # main()
    app = QApplication(sys.argv)
    ex = Capture_manager()
    sys.exit(app.exec_())

    # img = cv.imread('img/background/back_{}.png'.format(number))
    # plt.imshow(img)
    # plt.show()