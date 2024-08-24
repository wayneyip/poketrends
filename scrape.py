import pokebase as pb
import cv2
from urllib.request import urlopen, Request
from sklearn.cluster import KMeans 
import numpy as np
import matplotlib.pyplot as plt
import random
import webbrowser

# Get pokemon name and number
pokedex_number = random.randrange(1, 1025)
pokemon = pb.pokemon(pokedex_number)
print(pokedex_number, pokemon)

# Request image URL
img_url_request = Request(
	f'https://img.pokemondb.net/artwork/large/{pokemon}.jpg', 
	headers={'User-Agent': 'Chrome/23.0.1271.64'}
)
img_url_response = urlopen(img_url_request)

# Get image
img_webpage = img_url_response.read()
img_array = np.array(bytearray(img_webpage), dtype=np.uint8)
img = cv2.imdecode(img_array, -1)

# Convert image to pixel array
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pixels = img_rgb.reshape(-1, 3)

# Cluster colors
num_clusters = 5
kmeans = KMeans(n_clusters = num_clusters)
kmeans.fit(pixels)
dominant_colors = kmeans.cluster_centers_.astype(int)

for color in dominant_colors:
	print(color)

# Convert back
labels = kmeans.predict(pixels)
img_compressed = kmeans.cluster_centers_[labels].reshape(img.shape[0], img.shape[1], -1).astype(np.uint8)
plt.axis("off")
plt.title(f"{pokedex_number}_{pokemon}")
plt.imshow(img_compressed)
plt.show()

cv2.imshow(f"{pokedex_number}_{pokemon}", img)
cv2.waitKey(0)