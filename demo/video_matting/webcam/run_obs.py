import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import sys
sys.path.insert(1, '../pyKinectAzure/')
from pyKinectAzure import pyKinectAzure, _k4a

from src.models.modnet import MODNet

import argparse
import pyvirtualcam

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0, help="ID of webcam device (default: 0)")
parser.add_argument("--fps", action="store_true", help="output fps every second")
args = parser.parse_args()

#modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll' 


torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

print('Load pre-trained MODNet...')
pretrained_ckpt = './pretrained/modnet_webcam_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

GPU = True if torch.cuda.device_count() > 0 else False
if GPU:
    print('Use GPU...')
    modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
else:
    print('Use CPU...')
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))

modnet.eval()

print('Init WebCam...')
# Initialize the library with the path containing the module
#pyK4A = pyKinectAzure(modulePath)

# Open device
#pyK4A.device_open()

# Modify camera configuration
#device_config = pyK4A.config
#device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
#device_config.depth_mode = _k4a.K4A_DEPTH_MODE_OFF
#print(device_config)

# Start cameras using modified configuration
#pyK4A.device_start_cameras(device_config)
# Set up webcam capture.
vc = cv2.VideoCapture(args.camera)

# Check the image has been read correctly
if not vc.isOpened():
    raise RuntimeError('Could not open video source')

pref_width = 1280
pref_height = 720
pref_fps_in = 15
vc.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
vc.set(cv2.CAP_PROP_FPS, pref_fps_in)

# Query final capture device values (may be different from preferred settings).
width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = vc.get(cv2.CAP_PROP_FPS)
print(f'Webcam capture started ({width}x{height} @ {fps_in}fps)')

print('Start matting...')

fps_out = 24

try:
    with pyvirtualcam.Camera(width, height, fps_out, print_fps=args.fps) as cam:
        print(f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')
        # Get capture
        #pyK4A.device_get_capture()

        # Get the color image from the capture
        #color_image_handle = pyK4A.capture_get_color_image()

        # Read and convert the image data to numpy array:
        #frame_np = pyK4A.image_convert_to_numpy(color_image_handle)

        while True:
            # Read frame from webcam.
            ret, frame_np = vc.read()
            if not ret:
                raise RuntimeError('Error fetching frame')
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
            frame_np = frame_np[:, 120:792, :]
            frame_np = cv2.flip(frame_np, 1)

            frame_PIL = Image.fromarray(frame_np)
            frame_tensor = torch_transforms(frame_PIL)
            frame_tensor = frame_tensor[None, :, :, :]
            if GPU:
                frame_tensor = frame_tensor.cuda()
                
            with torch.no_grad():
                _, _, matte_tensor = modnet(frame_tensor, True)

            matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
            matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
            #fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
            fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
            #view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))
            view_np = np.uint8(fg_np)
            view_np = cv2.resize(view_np, (1280, 720), cv2.INTER_AREA)

            # Send to virtual cam.
            cam.send(view_np)

            # Wait until it's time for the next frame.
            cam.sleep_until_next_frame()
            #view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

finally:
    vc.release()
