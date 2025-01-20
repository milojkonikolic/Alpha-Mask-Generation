import tkinter as tk
import yaml

from object_extractor import ObjectExtractor


if __name__ == "__main__":

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    root = tk.Tk()
    app = ObjectExtractor(root, config["image_width"], config["image_height"], config["model_info"])
    root.mainloop()
