import torch
import sys
import DataProcessor
import matplotlib.pyplot as plt
import numpy as np

def load_model(model_path):
    model = torch.load(model_path)
    return model

dp = DataProcessor.DataProcessor()

if len(sys.argv) < 3:
    print("Usage: python ModelLoader.py <model_path> <data_path>")
    sys.exit()

model_path = sys.argv[1]
data_path = sys.argv[2]

sdf_shapes = dp.create_sdf_shapes(dp.read_data(data_path))
model = load_model(model_path)

for i in range(5):
    tensor_inputs = torch.FloatTensor(sdf_shapes[i]).reshape(1, 1, 32, 32)
    output = model(tensor_inputs)
    fig, axs = plt.subplots(2)
    axs[0].imshow(output[0][1].detach().numpy(), cmap='hot', interpolation='nearest')
    axs[1].imshow(tensor_inputs[0][0].detach().numpy(), cmap='hot', interpolation='nearest')
    plt.show()
    

