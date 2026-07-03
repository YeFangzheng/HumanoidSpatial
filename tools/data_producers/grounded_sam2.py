# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import cv2
import numpy as np
import torch
import supervision as sv

from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
)

GROUDED_SAM2_DIR = "/shared_disk/users/haoyu.wang/GroundedSAM2"
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"


class GroundedSAM2:
    def __init__(
        self,
        classes,
        color_palette=None,
        sam2_model_config=SAM2_MODEL_CONFIG,
        sam2_checkpoint=SAM2_CHECKPOINT,
        grounded_dino_config=GROUNDING_DINO_CONFIG,
        grounded_dino_checkpoint=GROUNDING_DINO_CHECKPOINT,
        box_threshold=0.25,
        text_threshold=0.0,
    ):  
        self.text_prompt = '.'.join(classes)
        self.classes = classes

        if color_palette:
            assert len(classes) == len(color_palette.colors)
            self.color_palette = color_palette
        else:
            self.color_palette = sv.ColorPalette.DEFAULT

        # self.sam_cfg = os.path.join(GROUDED_SAM2_DIR, sam2_model_config)
        self.sam2_cfg = sam2_model_config
        self.sam2_checkpoint = os.path.join(GROUDED_SAM2_DIR, sam2_checkpoint)
        # self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model = build_sam2(self.sam2_cfg, self.sam2_checkpoint, device="cuda")
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model, mask_threshold=-1.0)

        self.grounding_cfg = os.path.join(GROUDED_SAM2_DIR, grounded_dino_config)
        self.grounding_checkpoint = os.path.join(
            GROUDED_SAM2_DIR, grounded_dino_checkpoint
        )
        self.grounding_model = load_model(
            model_config_path=self.grounding_cfg,
            model_checkpoint_path=self.grounding_checkpoint,
            device="cuda",
        )

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(self, img_path, out_path):
        torch.autocast(device_type="cuda", dtype=torch.float32).__enter__()

        image_source, image = load_image(img_path)

        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            remove_combined=True,
        )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # FIXME: figure how does this influence the G-DINO model
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.sam2_predictor.set_image(image_source)
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()
        class_names = labels

        # class_ids = np.array(list(range(len(class_names))))
        class_ids = np.array([self.classes.index(name) for name in class_names])

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(class_names, confidences)
        ]

        """
        Visualize image with supervision useful API
        """

        img = cv2.imread(img_path)
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids,
        )

        box_annotator = sv.BoxAnnotator(color=self.color_palette)
        annotated_frame = box_annotator.annotate(
            scene=img.copy(), detections=detections
        )

        label_annotator = sv.LabelAnnotator(color=self.color_palette)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        mask_annotator = sv.MaskAnnotator(color=self.color_palette, opacity=0.3)
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        if not os.path.exists(os.path.dirname(out_path)) and os.path.dirname(out_path) != '':
            os.makedirs(os.path.dirname(out_path))
        cv2.imwrite(out_path, annotated_frame)

        labels = np.ones((masks.shape[1], masks.shape[2]), dtype=np.int32) * 255
        for detection_idx in np.flip(np.argsort(detections.area)):
            mask = detections.mask[detection_idx]
            labels[mask] = class_ids[detection_idx]

        # masks = torch.tensor(masks * np.array(confidences)[..., None, None])
        # scores, idx = masks.max(0)
        # labels = class_ids[idx]
        # labels[scores == 0] = 255

        # cv2.imwrite('mask.jpg', masks.any(0).numpy().astype(np.uint8) * 255)
        
        return labels


if __name__ == "__main__":
    classes = ['car', 'pedestrian', 'large vehicle', 'cyclist',
               'tree', 'bush', 'pole', 'cone', 'traffic light', 'fence', 'building',
               'road', 'curb', 'sidewalk', 'grassland']
    
    color_palette = sv.ColorPalette.from_hex([
        '#0000ff', '#191970', '#9370db', '#ffb6c1',
        '#008000', '#008000', '#ffff00', '#808080', '#d3d3d3', '#ff8c00', '#ffdead',
        '#ffffff', '#8b0000', '#ff0000', '#90ee90'])

    # img_path = "./data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"
    img_path = "./data/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402928112460.jpg"
    groundedsam2 = GroundedSAM2(classes, color_palette=color_palette)

    groundedsam2.predict(img_path, 'demo.jpg')
