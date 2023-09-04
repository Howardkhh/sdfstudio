#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from torch import Tensor
import time

import numpy as np
import torch
import tyro
import matplotlib.pyplot as plt

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs import base_config as cfg # apparently this line is needed, or else circular import will occur
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.viewer.server.utils import three_js_perspective_camera_focal_length

import argparse

file_counter = 0

# copied from nerfstudio
def apply_pca_colormap(image):
    """Convert feature image to 3-channel RGB via PCA. The first three principle
    components are used for the color channels, with outlier rejection per-channel

    Args:
        image: image of arbitrary vectors

    Returns:
        Tensor: Colored image
    """
    original_shape = image.shape
    image = image.view(-1, image.shape[-1])
    _, _, v = torch.pca_lowrank(image)
    image = torch.matmul(image, v[..., :3])
    d = torch.abs(image - torch.median(image, dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    m = 3.0  # this is a hyperparam controlling how many std dev outside for outliers
    rins = image[s[:, 0] < m, 0]
    gins = image[s[:, 1] < m, 1]
    bins = image[s[:, 2] < m, 2]

    image[:, 0] -= rins.min()
    image[:, 1] -= gins.min()
    image[:, 2] -= bins.min()

    image[:, 0] /= rins.max() - rins.min()
    image[:, 1] /= gins.max() - gins.min()
    image[:, 2] /= bins.max() - bins.min()

    image = torch.clamp(image, 0, 1)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return image.view(*original_shape[:-1], 3)

def render_img(
    pipeline: Pipeline,
    cameras: Cameras,
    rendered_output_names: List[str],
    rendered_resolution_scaling_factor: float = 1.0,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
) -> List[Tensor]:

    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    camera_ray_bundle = cameras.generate_rays(camera_indices=0)

    start_time = time.time()
    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
    CONSOLE.print(f"Image rendered in {time.time() - start_time:.2f} seconds")

    ############# Save rendered images for visualization #############
    global file_counter

    for rendered_output_name in outputs.keys(): # rendered_output_names:
        output_image = outputs[rendered_output_name].cpu().numpy()
        if np.min(output_image) < 0 or np.max(output_image) > 1:
            output_image = (output_image - np.min(output_image)) / (np.max(output_image) - np.min(output_image))
        if output_image.shape[-1] == 3:
            plt.imsave(f"tmp/{file_counter}_{rendered_output_name}.jpg", output_image)
        elif output_image.shape[-1] == 1:
            plt.imsave(f"tmp/{file_counter}_{rendered_output_name}.jpg", output_image[..., 0], cmap="gray")
        else:
            output_image = apply_pca_colormap(outputs[rendered_output_name]).cpu().numpy()
            plt.imsave(f"tmp/{file_counter}_{rendered_output_name}.jpg", output_image)
            # CONSOLE.print(f"Skipping {rendered_output_name} as it is not an image. (shape: {output_image.shape})")

    file_counter += 1
    ##################################################################

    return [outputs[key] for key in rendered_output_names]


class RenderImages():
    def __init__(self, args):
        self.load_config = Path(args.config)
        self.downscale_factor: float = 1.0
        self.eval_num_rays_per_chunk: Optional[int] = None
        self.rendered_output_names: List[str] = ["rgb"]
        self.depth_near_plane: Optional[float] = None
        self.depth_far_plane: Optional[float] = None
        
        _, self.pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        self.image_height = args.image_height
        self.image_width = args.image_width
        fov = args.fov
        self.focal_length = three_js_perspective_camera_focal_length(fov, self.image_height)

    def execute(self) -> List[Tensor]:
        
        # get input
        c2w = torch.tensor(eval(input("c2w matrix: "))).view(4, 4)[:3]
        # print(c2w)
        camera_path = Cameras(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=self.image_width / 2,
            cy=self.image_height / 2,
            camera_to_worlds=c2w,
        )

        return render_img(
            self.pipeline,
            camera_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
        )


def main_func():
    parser = argparse.ArgumentParser(description='Render Images')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--image-width', help='image width', default=1920, type=int)
    parser.add_argument('--image-height', help='image height', default=1080, type=int)
    parser.add_argument('--fov', help='field of view', default=50, type=float)
    args = parser.parse_args()
    
    ri = RenderImages(args)
    while True:
        ri.execute()

if __name__ == "__main__":
    main_func()