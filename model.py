import numpy as np
from segment_anything import SamPredictor, sam_model_registry


class SAMModel:
    """A wrapper class for the Segment Anything Model (SAM).
    This class provides an interface to initialize and use the SAM model for
    image segmentation tasks. It handles model loading and prediction.
    Attributes:
        model: The loaded SAM model instance.
        predictor: A SamPredictor instance for making predictions.
    """

    def __init__(self, model_type: str = "vit_h", 
                 checkpoint_path: str = "./checkpoints/sam_vit_h_4b8939.pth"):
        """Initializes the SAM model.
        Args:
            model_type: The type of SAM model to use. Defaults to "vit_h".
            checkpoint_path: Path to the model checkpoint file. Defaults to 
                "sam_vit_h_4b8939.pth".
        """
        self.model = sam_model_registry[model_type](checkpoint_path)
        self.predictor = SamPredictor(self.model)

    def predict(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Predicts segmentation mask for an object in the given image.
        Args:
            image: Input image as a numpy array.
            boxes: Bounding box coordinates as a numpy array in format [x1, y1, x2, y2].
        Returns:
            A binary mask as a 2D numpy array where 1 indicates the object region
            and 0 indicates the background.
        """
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes[None, :],
            multimask_output=True,
        )
        mask = masks[-1]
        h, w = mask.shape[-2:]
        mask = np.squeeze(mask.reshape(h, w, 1))
        return mask
