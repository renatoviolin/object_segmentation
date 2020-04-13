
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys
from argparse import Namespace
import numpy as np
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from centermask.predictor import VisualizationDemo
from centermask.config import get_cfg


args = Namespace(input='./people/',
                 output='results',
                 config_file='../configs/centermask_V_99_eSE_FPN_ms_3x.yaml',
                 opts=['MODEL.WEIGHTS', '../pretrained_models/centermask2-V-99-eSE-FPN-ms-3x.pth'],
                 confidence_threshold=0.5)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


mp.set_start_method("spawn", force=True)
cfg = setup_cfg(args)
demo = VisualizationDemo(cfg)


def generate_segments(input_path, output_folder):
    file_id = str(uuid.uuid1()).split('-')[0]
    out = os.path.join(output_folder, f'center_{file_id}.png')

    img = read_image(input_path, format="BGR")
    predictions, visualized_output = demo.run_on_image(img)
    visualized_output.save(out)
    return out
