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
	c=opaque_pixels_rgb,
	s=100
)
ax1.set_xlabel('Lightness')
ax1.set_ylabel('A: Red-Green')
ax1.set_zlabel('B: Blue-Yellow')

# Figure 4: compressed colors 3D graph
fig = plt.figure()
ax2 = fig.add_subplot(projection='3d')
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
fig = plt.figure()
ax3 = fig.add_subplot()
ax3.scatter(
	list(range(1, 7)),
	inertias
)
ax3.set_xlabel('Number of clusters')
ax3.set_ylabel('Inertia')
	

plt.show()