import pokebase as pb
import cv2
from urllib.request import urlopen, Request
from sklearn.cluster import KMeans 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import math
import random
import webbrowser
import os

os.environ['OMP_NUM_THREADS'] = '6'

# Get pokemon name and number
pokedex_number = random.randrange(1, 1025)
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
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

scale_factor = 5

# Get list of opaque pixels only
opaque_mask = img[:, :, 3] != 0 
opaque_pixels = img_lab[opaque_mask][:, :3]
opaque_pixels[:, 0] = img_lab[opaque_mask][:, 0] / scale_factor 
opaque_pixels_rgb = img_rgb[opaque_mask][:, :3] / 255

def find_elbow(data):

	inertias = []

	for i in range(1, 7):
		test_kmeans = KMeans(n_clusters=i).fit(data)
		test_kmeans.fit(opaque_pixels)
		inertias.append(test_kmeans.inertia_)

	def normalize(arr):
		min_num = np.min(arr)
		num_range = np.max(arr) - np.min(arr)
		normalized_arr = (arr - min_num) / num_range * 5
		return normalized_arr

	normalized_inertias = normalize(inertias)

	for i in range(len(inertias)):
		if i >= 1:
			x_diff = 1
			y_diff = normalized_inertias[i] - normalized_inertias[i-1]
			angle = np.arctan2(y_diff, x_diff)
			print(angle)
			if angle > -0.7854: # 45 degrees, or PI/4 radians
				print(f"found elbow: {i}")
				return i, normalized_inertias

# Cluster the opaque pixels' colors
num_clusters, inertias = find_elbow(opaque_pixels)
kmeans = KMeans(n_clusters = num_clusters)
kmeans.fit(opaque_pixels)
reduced_lab = kmeans.cluster_centers_.astype(np.uint8)

# Convert clustered colors to RGB (for chart display)
reduced_lab[:, 0] = reduced_lab[:, 0] * scale_factor
reduced_lab_reshaped = reduced_lab.reshape(1, -1, 3)
reduced_rgb_reshaped = cv2.cvtColor(reduced_lab_reshaped, cv2.COLOR_LAB2RGB)
reduced_rgb = reduced_rgb_reshaped.reshape(-1, 3)
reduced_rgb_normalized = reduced_rgb / 255
reduced_rgb_hex = [rgb2hex(color) for color in reduced_rgb_normalized]

# Reconstruct image with clustered colors
labels = kmeans.labels_
img_compressed = img.copy()
img_compressed[opaque_mask, :3] = reduced_lab[labels]
img_compressed = cv2.cvtColor(img_compressed[:, :, :3], cv2.COLOR_LAB2RGB)

# Get frequency of each color
unique_labels, counts = np.unique(labels, return_counts=True)
total_pixels = len(labels)
frequencies = counts / total_pixels

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Figure 1: original image
axs[0, 0].imshow(img_rgb)
axs[0, 0].axis('off')
axs[0, 0].set_title(f"{pokedex_number}_{pokemon}")

# Figure 2: compressed image
axs[0, 1].imshow(img_compressed)
axs[0, 1].axis('off')
axs[0, 1].set_title(f"{pokedex_number}_{pokemon} (Compressed)")
 
# Figure 3: compressed colors
axs[0, 2].pie(
	frequencies, 
	colors=reduced_rgb_normalized, 
	labels=reduced_rgb_hex, 
	labeldistance=0.5
)
axs[0, 2].set_title(f"{pokedex_number}_{pokemon} Color Distribution")

# Figure 4: uncompressed colors 3D graph
ax1 = fig.add_subplot(2, 3, 4, projection='3d')
ax1.scatter(
	opaque_pixels[:, 0], 
	opaque_pixels[:, 1], 
	opaque_pixels[:, 2], 
	c=opaque_pixels_rgb,
	s=100
)
ax1.set_xlabel('Lightness')
ax1.set_ylabel('A: Red-Green')
ax1.set_zlabel('B: Blue-Yellow')

# Figure 5: compressed colors 3D graph
ax2 = fig.add_subplot(2, 3, 5, projection='3d')
ax2.scatter(
	opaque_pixels[:, 0], 
	opaque_pixels[:, 1], 
	opaque_pixels[:, 2], 
	c=reduced_rgb_normalized[labels],
	s=100
)
ax2.set_xlabel('Lightness')
ax2.set_ylabel('A: Red-Green')
ax2.set_zlabel('B: Blue-Yellow')

# Figure 6: elbow graph
ax3 = fig.add_subplot(2, 3, 6)
ax3.scatter(
	list(range(1, 7)),
	inertias
)
ax3.set_xlabel('Number of clusters')
ax3.set_ylabel('Inertia')

for ax in axs[1, :]:
    ax.set_axis_off()

plt.show()