import tensorflow as tf
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import numpy as np
from PIL import Image
import cv2
import uuid

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef().FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


MODEL = DeepLabModel('../pretrained_models/deeplabv3_pascal_trainval_2018_01_04.tar.gz')


def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r_all = np.zeros_like(image).astype(np.uint8)
    g_all = np.zeros_like(image).astype(np.uint8)
    b_all = np.zeros_like(image).astype(np.uint8)
    maps = []

    for l in range(nc):
        idx = image == l
        r_all[idx] = label_colors[l, 0]
        g_all[idx] = label_colors[l, 1]
        b_all[idx] = label_colors[l, 2]

        r_unique = np.zeros_like(image).astype(np.uint8)
        g_unique = np.zeros_like(image).astype(np.uint8)
        b_unique = np.zeros_like(image).astype(np.uint8)
        r_unique[idx] = label_colors[l, 0]
        g_unique[idx] = label_colors[l, 1]
        b_unique[idx] = label_colors[l, 2]

        maps.append(np.stack([r_unique, g_unique, b_unique], axis=2))

    rgb = np.stack([r_all, g_all, b_all], axis=2)
    return rgb, maps


def segment(path):
    original_im = Image.open(path)
    resized_im, seg_map = MODEL.run(original_im)
    mask, all_maps = decode_segmap(seg_map)
    return mask, resized_im, all_maps, np.unique(seg_map)


def generate_segments(input_path, output_folder):
    mask, img, all_maps, class_idx = segment(input_path)
    out_files = []
    for i in class_idx:
        file_id = str(uuid.uuid1()).split('-')[0]
        out = os.path.join(output_folder, f'{i}_{file_id}.png')
        out_files.append(out)
        if i == 0:
            added_image = cv2.addWeighted(np.array(img), 0.5, mask, 1.3, 0)
            cv2.imwrite(out, cv2.cvtColor(added_image, cv2.COLOR_BGR2RGB))
        else:
            added_image = cv2.addWeighted(np.array(img), 0.5, all_maps[i], 1.3, 0)
            cv2.imwrite(out, cv2.cvtColor(added_image, cv2.COLOR_BGR2RGB))
    return out_files, list(LABEL_NAMES[class_idx])
