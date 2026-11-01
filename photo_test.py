import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from core.data import load_images



train = 'train'
frames = range(0, 600, 10) 
images = load_images(train)


fig, axes = plt.subplots(4,5, figsize = (12,6))


for axe, i in zip(axes.flat,frames):
    picture = Image.open(io.BytesIO(images[i]))
    axe.imshow(picture)
    axe.set_title(f"frame number {i}")
    axe.axis('off')
fig.suptitle("20 Images from one ride")
plt.savefig("images/image.png")
plt.show()






