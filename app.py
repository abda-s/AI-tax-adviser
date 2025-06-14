# app.py
from PyQt5.QtWidgets import QApplication
from gui.pyqt_app import SmartTaxAdvisor
import sys

def main():
    app = QApplication(sys.argv)
    window = SmartTaxAdvisor()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()