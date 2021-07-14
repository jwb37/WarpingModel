from ImageWarpNet import ImageWarpNet
from PIL import Image

imgA = Image.open('../inputs/clock.png')
imgB = Image.open('../inputs/dali_melting_clock.png')

WarpNet = ImageWarpNet()

out_img = WarpNet.warp(imgA, imgB)
out_img.save('output.png')
