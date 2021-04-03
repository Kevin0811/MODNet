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

modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll' 


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
pyK4A = pyKinectAzure(modulePath)

# Open device
pyK4A.device_open()

# Modify camera configuration
device_config = pyK4A.config
device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
device_config.depth_mode = _k4a.K4A_DEPTH_MODE_OFF
print(device_config)

# Start cameras using modified configuration
pyK4A.device_start_cameras(device_config)

print('Start matting...')
k=0
while(True):
    # Get capture
    pyK4A.device_get_capture()

	# Get the color image from the capture
    color_image_handle = pyK4A.capture_get_color_image()

	# Check the image has been read correctly
    if color_image_handle:
        
        # Read and convert the image data to numpy array:
        frame_np = pyK4A.image_convert_to_numpy(color_image_handle)
        
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
        fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
        view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))
        view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

        cv2.imshow('MODNet - WebCam [Press \'Q\' To Exit]', view_np)
        k = cv2.waitKey(1)

        pyK4A.image_release(color_image_handle)
    
    pyK4A.capture_release()

    if k==27:
        break

print('Exit...')
pyK4A.device_stop_cameras()
pyK4A.device_close()