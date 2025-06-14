from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
import sys

def main():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle('PyQt5 Test')
    window.setGeometry(100, 100, 300, 200)
    
    # Create central widget and layout
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)
    
    # Add a test button
    button = QPushButton('Test Button')
    layout.addWidget(button)
    
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 