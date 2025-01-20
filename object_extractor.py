import logging
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

from model import get_model


def get_logger():
    """Gets a logger instance for the application."""
    logger = logging.getLogger("Alpha Mask")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s]: %(message)s ")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


class ObjectExtractor:
    """A GUI application for extracting objects from images and placing them on virtual backgrounds.
    This class provides functionality to load images, select objects via bounding boxes,
    extract them using SAM model, and overlay them on virtual backgrounds.
    """

    def __init__(self, root, img_width, img_height, model_name):
        """Initializes the ObjectExtractor application.
        Args:
            root: The root tkinter window that will contain this application.
        """
        self.root = root
        self.root.title("Object Extractor App")
        self.logger = get_logger()

        # Set canvas dimensions
        self.canvas_shape = (img_width, img_height)
        self.canvas = tk.Canvas(root, width=self.canvas_shape[0], height=self.canvas_shape[1], bg="gray")
        self.canvas.pack(fill="both", expand=True)

        self.model = get_model(model_name, self.logger)

        self.image = None
        self.image_path = None
        self.virtual_image = None
        self.virtual_image_path = None
        self.mask = None
        self.rect_id = None
        self.start_x = self.start_y = self.end_x = self.end_y = 0
        self.dot_x = self.dot_y = None

        # Add scale slider
        self.scale = tk.DoubleVar(value=1.0)
        self.scale_label = tk.Label(root, text="Scale:")
        self.scale_label.pack(side="left", padx=5, pady=10)
        
        self.scale_slider = tk.Scale(root, from_=0.1, to=5.0, resolution=0.05, orient=tk.HORIZONTAL, 
                                     variable=self.scale, length=300)
        self.scale_slider.pack(side="left", padx=5, pady=10)

        self.load_image_button = tk.Button(root, text="Load Object Image", command=self.load_object_image)
        self.load_image_button.pack(side="left", padx=10, pady=10)

        self.virtual_image_button = tk.Button(root, text="Load Virtual Background", command=self.load_virtual_image)
        self.virtual_image_button.pack(side="left", padx=10, pady=10)

        self.extract_button = tk.Button(root, text="Extract and Place Object", command=self.extract_and_place)
        self.extract_button.pack(side="left", padx=10, pady=10)

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

    def load_object_image(self):
        """Loads and displays the source image containing the object to extract.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # Apply scaling to the coordinates
            scale_factor = self.scale.get()
            self.scale_image(scale_factor)
            self.image = cv2.resize(self.image, self.canvas_shape)
            self.display_image(self.image)

    def scale_image(self, scale_factor=1.0, rotation_angle=0.0):
        img_center = np.array(self.image.shape[:2])[::-1] / 2
        zoom_matrix = cv2.getRotationMatrix2D(tuple(img_center), rotation_angle, scale_factor)
        img_shape = (self.image.shape[1], self.image.shape[0])
        self.image = cv2.warpAffine(self.image, zoom_matrix, img_shape)

    def load_virtual_image(self):
        """Loads and displays the virtual background image.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.virtual_image_path = file_path
            self.virtual_image = cv2.imread(file_path)
            self.virtual_image = cv2.cvtColor(self.virtual_image, cv2.COLOR_BGR2RGB)
            self.virtual_image = cv2.resize(self.virtual_image, self.canvas_shape)
            self.display_image(self.virtual_image)

    def display_image(self, image):
        """Displays an image on the canvas.
        Args:
            image: A numpy array representing the image to display.
        """
        display_image = cv2.resize(image, self.canvas_shape)
        display_image = Image.fromarray(display_image.astype('uint8'))
        photo = ImageTk.PhotoImage(display_image)
        # Update canvas
        self.canvas.delete("all")
        self.canvas.image = photo
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def start_draw(self, event):
        """Handles mouse button press events for drawing or positioning.
        If no virtual image is loaded, starts drawing a bounding box.
        If virtual image is loaded, records click position for object placement.
        Args:
            event: A tkinter event object containing the mouse coordinates.
        """
        if self.virtual_image is None:
            self.logger.info("Draw bbox around the object that you want to put on the result image")
            self.start_x, self.start_y = event.x, event.y
            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")
            self.logger.info(f"Start coords: {self.start_x}, {self.start_y}")
        else:
            self.logger.info("Click on the image where you want to put the extracted object")
            self.virtual_position_x, self.virtual_position_y = event.x, event.y
            self.logger.info(f"Position on the virtual image: {self.virtual_position_x}, {self.virtual_position_y}")

    def draw_rect(self, event):
        """Updates the bounding box during mouse drag.
        Args:
            event: A tkinter event object containing the current mouse coordinates.
        """
        if self.virtual_image is None:
            self.end_x, self.end_y = event.x, event.y
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, self.end_x, self.end_y)

    def end_draw(self, event):
        """Handles mouse button release events for bounding box drawing.
        Args:
            event: A tkinter event object containing the final mouse coordinates.
        """
        if self.virtual_image is None:
            self.end_x, self.end_y = event.x, event.y
            self.logger.info(f"End coords: {self.end_x}, {self.end_y}")

    def extract_and_place(self):
        """Extracts the selected object using SAM model and initiates placement.
        """
        if self.image is None or not (self.start_x and self.start_y and self.end_x and self.end_y):
            self.logger.error("No object image or bounding box provided.")
            return

        # Extract object using bounding box
        input_box = np.array([self.start_x, self.start_y, self.end_x, self.end_y])
        self.mask = self.model.predict(self.image, input_box)

        extracted_object = cv2.bitwise_and(self.image, self.image, mask=self.mask.astype(np.uint8))
        self.overlay_object_on_virtual()

    def overlay_object_on_virtual(self):
        """Overlays the extracted object onto the virtual background.
        Applies the mask to blend the extracted object with the virtual background
        using alpha compositing.
        """
        # Place the object onto the virtual image
        shift_x, shift_y = self.position_mask()
        self.shift_image(shift_x, shift_y)

        # Apply Gaussian blur to smooth the mask edges
        smoothed_mask = cv2.GaussianBlur(self.mask.astype(np.float32), (3, 3), 3)
        alpha_mask = np.stack([smoothed_mask, smoothed_mask, smoothed_mask], axis=2).astype(np.float32)
        cv2.imwrite("./examples/alpha_mask.png", (self.mask * 255).astype(np.uint8))

        foreground = cv2.multiply(alpha_mask, self.image.astype(np.float32))
        background = cv2.multiply(1 - alpha_mask, self.virtual_image.astype(np.float32))
        result_image = cv2.add(foreground, background).astype(np.uint8)

        result_image = self.refine_edges(result_image, smoothed_mask)

        cv2.imwrite("./examples/result_image.png", result_image)
        self.display_image(result_image)
    
    def refine_edges(self, image, mask, blur_amount=1):
        """Refines the edges of the composited image for smoother blending.
        Args:
            image: The composited image
            mask: The alpha mask
            blur_amount: Amount of edge blur to apply
        Returns:
            The image with refined edges
        """
        # Create a narrow band around the mask edges
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)

        # Apply selective blurring only to the edge regions
        blurred = cv2.GaussianBlur(image, (3, 3), blur_amount)
        result = image.copy()
        edge_mask = np.stack([edge_mask, edge_mask, edge_mask], axis=2)
        result = np.where(edge_mask > 0, blurred, result)

        return result

    def position_mask(self):
        """Calculates and applies the shift needed to position the mask at the clicked location.
        Returns:
            tuple: A pair of integers (shift_x, shift_y) representing the translation needed.
        """
        y_indices, x_indices = np.where(self.mask)

        current_center_x = np.mean(x_indices)
        current_center_y = np.mean(y_indices)

        # Calculate shift needed
        shift_x = int(self.virtual_position_x - current_center_x)
        shift_y = int(self.virtual_position_y - current_center_y)
        
        # Create new mask of same size
        new_mask = np.zeros_like(self.mask)
        
        # Shift mask contents
        for y, x in zip(y_indices, x_indices):
            new_y = y + shift_y
            new_x = x + shift_x
            
            # Only set pixel if it's within bounds
            if (0 <= new_y < self.mask.shape[0] and 
                0 <= new_x < self.mask.shape[1]):
                new_mask[new_y, new_x] = True
        self.mask = new_mask
        return shift_x, shift_y

    def shift_image(self, shift_x, shift_y):
        """Applies a translation to the source image.
        Args:
            shift_x: Integer number of pixels to shift horizontally.
            shift_y: Integer number of pixels to shift vertically.
        """

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        # Apply the translation to the image
        shifted_image = cv2.warpAffine(
            self.image, 
            M, 
            self.canvas_shape,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        self.image = shifted_image
