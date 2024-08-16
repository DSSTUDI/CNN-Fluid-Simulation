import torch
import DataProcessor
import matplotlib.pyplot as plt
import numpy as np
import config

def load_model(model_path):
    model = torch.load(model_path)
    return model

dp = DataProcessor.DataProcessor()

model_path = config.config.model_path
shape_data_path = config.config.shape_data_path
example_no = config.config.examples
example_path = config.config.example_path
data_path = config.config.dataset_path

clean_horizontals = dp.read_data(f'{data_path}/horizontal')
clean_verticals = dp.read_data(f'{data_path}/vertical')
total_velocities = dp.calculate_total_velocities(clean_horizontals, clean_verticals)
sdf_shapes = dp.create_sdf_shapes(dp.read_data(shape_data_path))
model = load_model(model_path)

for i in range(example_no):
    tensor_inputs = torch.FloatTensor(sdf_shapes[i]).reshape(1, 1, sdf_shapes[i].shape[0], sdf_shapes[i].shape[0])
    output = model(tensor_inputs)
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(output[0][1].detach().numpy(), cmap='hot', interpolation='nearest')
    axs[0].set_title('Model Output')
    axs[1].imshow(tensor_inputs[0][0].detach().numpy(), cmap='hot', interpolation='nearest')
    axs[1].set_title('Model Input')
    axs[2].imshow(total_velocities[i], cmap='hot', interpolation='nearest')
    axs[2].set_title('Simulation Output')
    plt.savefig(f'{example_path}/{model.__class__.__name__}_{i}.png')
    

