# app.py
from gui.gui_app import GUIApp
import tkinter as tk

def main():
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()