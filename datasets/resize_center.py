import numpy as np
from PIL import Image

final_width = 384
final_height = 384

def img_resize(img):
	w, h = img.size
	ar = float(w)/float(h)
	if w/final_width<h/final_height:
		new_w = final_width
		new_h = int(new_w/ar)
		a = new_h - final_height
		resize_img = img.resize((new_w, new_h))
		final_image = resize_img.crop((0,a/2,final_width,(a/2+final_height)))
	elif w/final_width>h/final_height:
		new_h = final_height
		new_w = int(new_h*ar)
		a = new_w - final_width
		resize_img = img.resize((new_w, new_h))
		final_image = resize_img.crop((a/2,0,a/2+final_width,final_height))
	else:
		resize_img = img.resize((final_width, final_height))
		final_image = resize_img
	return final_image