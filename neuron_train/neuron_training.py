import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import os
# from raybnn import RayBNN
from scipy.spatial import distance

# =========================================================================================================
# load image
image_path = "C:\Users\Joseph\Desktop\neuron_train\test.png"



img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not load image. Check the file path.")
else:
    print("Image loaded successfully.")

# Gaussian denoising
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# enhanced contrast
img_eq = cv2.equalizeHist(img_blur)

# making neurons clearer
_, binary = cv2.threshold(img_eq, 100, 255, cv2.THRESH_BINARY)

# show optimized image
plt.figure(figsize=(12,4))
plt.subplot(1,3,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(1,3,2), plt.imshow(img_eq, cmap='gray'), plt.title("Equalized")
plt.subplot(1,3,3), plt.imshow(binary, cmap='gray'), plt.title("Binarized")
plt.show()

# draw contour
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_contours = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

# 标记神经元中心点
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(img_contours, (cx, cy), 3, (255, 0, 0), -1)  # 用蓝色标记中心点

# display result
plt.figure(figsize=(6,6))
plt.imshow(img_contours)
plt.title("Neuron Contours with Centers")
plt.axis("off")
plt.show()

# =========================================================================================================

# Build Neural Topology
G = nx.Graph()

# Add Neuron Nodes
for i, contour in enumerate(contours):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])  # compute centre
        cy = int(M["m01"] / M["m00"])
        G.add_node(i, pos=(cx, cy))  # add neuron

# compute spatial connection
nodes = list(G.nodes)
for i in range(len(nodes)):
    for j in range(i+1, len(nodes)):
        dist = distance.euclidean(G.nodes[nodes[i]]['pos'], G.nodes[nodes[j]]['pos'])
        if dist < 150:
            G.add_edge(nodes[i], nodes[j], weight=dist)

# show topology
plt.figure(figsize=(6,6))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=300, node_color="red")
plt.title("Neuron Network Graph")
plt.show()

# 在原始图像上标记神经元节点
img_with_nodes = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
for node in nodes:
    cx, cy = G.nodes[node]['pos']
    cv2.circle(img_with_nodes, (cx, cy), 3, (255, 0, 0), -1)  # 用蓝色标记节点

# display result
plt.figure(figsize=(6,6))
plt.imshow(img_with_nodes)
plt.title("Neuron Network Nodes on Image")
plt.axis("off")
plt.show()

# =========================================================================================================
# Convert to Pytorch!
# Graph -> Tensor
adj_matrix = nx.to_numpy_array(G)
tensor_input = torch.tensor(adj_matrix, dtype=torch.float32)

# Define the save path
save_dir = "C:/Users/Joseph/Desktop/raybnn/neuron_train/data"
save_path = os.path.join(save_dir, "new_neuron_graph_tensor.pt")

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Save Tensor data
torch.save(tensor_input, save_path)

print("Converted to Neural Network，saved as PyTorch Data！")

# =========================================================================================================
# Train RayBNN - just structed
# class NeuronRayBNN(nn.Module):
#     def __init__(self, input_size):
#         super(NeuronRayBNN, self).__init__()
#         self.raybnn = RayBNN(input_size, hidden_size=64, output_size=input_size)
    
#     def forward(self, x):
#         return self.raybnn(x)

# # read Topology Data
# num_neurons = tensor_input.shape[0]  # get number of neuron
# model = NeuronRayBNN(num_neurons)

# # loss function & optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# # Training
# epochs = 100
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     output = model(tensor_input)
#     loss = criterion(output, tensor_input)
#     loss.backward()
#     optimizer.step()

#     if epoch % 10 == 0:
#         print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# # save the training result
# torch.save(model.state_dict(), "/mnt/data/trained_raybnn.pth")
# print("Work Done！RayBNN Model has been Saved!")

# =========================================================================================================
# Generate Wave Pattern

# Reasoning Mode and Predict Neural Network's Wave Pattern
# model.eval()  
# wave_output = model(tensor_input)


# =========================================================================================================
# Convert to DLP Projection Data
# wave_pattern = wave_output.detach().numpy()  # convert to NumPy array
# wave_pattern = (wave_pattern - wave_pattern.min()) / (wave_pattern.max() - wave_pattern.min())  # normalization

# # Generate projection image
# projector_image = (wave_pattern * 255).astype(np.uint8)
# cv2.imwrite("/mnt/data/projector_wave.pgm", projector_image)

# print("Work Done! Wave Pattern has been converted to projection image！")