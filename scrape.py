import pokebase as pb
import cv2
from urllib.request import urlopen, Request
from sklearn.cluster import KMeans 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import random
import webbrowser
import os

os.environ['OMP_NUM_THREADS'] = '6'

# Get pokemon name and number
pokedex_number = 121 #random.randrange(1, 1025)
pokemon = pb.pokemon(pokedex_number)
print(pokedex_number, pokemon)

sprite_url = pokemon.sprites.front_default

img_url_request = Request(
	sprite_url, 
	headers={'User-Agent': 'Chrome/23.0.1271.64'}
)
img_url_response = urlopen(img_url_request)

img_webpage = img_url_response.read()
img_array = np.array(bytearray(img_webpage), dtype=np.uint8)
img = cv2.imdecode(img_array, -1)

# Turn all transparent pixels black
img[img[:, :, 3] == 0] = np.zeros(4)

# Convert image to pixel array
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Get list of opaque pixels only
opaque_mask = img[:, :, 3] != 0 
opaque_pixels = img_hsv[opaque_mask][:, :3]
opaque_pixels_rgb = img_rgb[opaque_mask][:, :3] / 255

# Cluster the opaque pixels' colors
num_clusters = 5
kmeans = KMeans(n_clusters = num_clusters)
kmeans.fit(opaque_pixels)
reduced_hsv = kmeans.cluster_centers_.astype(np.uint8)

# Convert clustered colors to RGB (for chart display)
reduced_hsv_reshaped = reduced_hsv.reshape(1, -1, 3)
reduced_rgb_reshaped = cv2.cvtColor(reduced_hsv_reshaped, cv2.COLOR_HSV2RGB)
reduced_rgb = reduced_rgb_reshaped.reshape(-1, 3)
reduced_rgb_normalized = reduced_rgb / 255
reduced_rgb_hex = [rgb2hex(color) for color in reduced_rgb_normalized]

# Reconstruct image with clustered colors
labels = kmeans.labels_
img_compressed = img.copy()
img_compressed[opaque_mask, :3] = reduced_hsv[labels]
img_compressed = cv2.cvtColor(img_compressed[:, :, :3], cv2.COLOR_HSV2RGB)

# Get frequency of each color
unique_labels, counts = np.unique(labels, return_counts=True)
total_pixels = len(labels)
frequencies = counts / total_pixels

# Figure 1: original image
plt.figure(1)
plt.axis("off")
plt.title(f"{pokedex_number}_{pokemon}")
plt.imshow(img_rgb)

# Figure 2: compressed image
plt.figure(2)
plt.axis("off")
plt.title(f"{pokedex_number}_{pokemon} (Compressed)")
plt.imshow(img_compressed)
 
# Figure 3: compressed colors
plt.figure(3)
plt.title(f"{pokedex_number}_{pokemon} Color Distribution")
plt.pie(
	frequencies, 
	colors=reduced_rgb_normalized, 
	labels=reduced_rgb_hex, 
	labeldistance=0.5
)

# Figure 4: uncompressed colors 3D graph
fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
ax1.scatter(
	opaque_pixels[:, 0], 
	opaque_pixels[:, 1], 
	opaque_pixels[:, 2], 
	c=opaque_pixels_rgb
)
ax1.set_xlabel('Hue')
ax1.set_ylabel('Saturation')
ax1.set_zlabel('Value')

# Figure 4: compressed colors 3D graph
fig = plt.figure()
ax2 = fig.add_subplot(projection='3d')
ax2.scatter(
	opaque_pixels[:, 0], 
	opaque_pixels[:, 1], 
	opaque_pixels[:, 2], 
	c=reduced_rgb_normalized[labels]
)
ax2.set_xlabel('Hue')
ax2.set_ylabel('Saturation')
ax2.set_zlabel('Value')

plt.show()