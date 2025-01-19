import tkinter as tk

from object_extractor import ObjectExtractor


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectExtractor(root)
    root.mainloop()
