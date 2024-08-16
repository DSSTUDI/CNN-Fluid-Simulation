import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import DataProcessor
import CNN32_Model
import CNN64_Model
import config

# initialise variables
model_type = config.config.model_type
epochs = config.config.epochs
test_train_split = config.config.test_train_split
learning_rate = config.config.learning_rate
model_name = config.config.model_name
data_path = config.config.dataset_path

dp = DataProcessor.DataProcessor()

# Load the data
sdf_shapes = dp.create_sdf_shapes(dp.read_data(f'{data_path}/shape'))
clean_horizontals = dp.read_data(f'{data_path}/horizontal')
clean_verticals = dp.read_data(f'{data_path}/vertical')
total_velocities = dp.calculate_total_velocities(clean_horizontals, clean_verticals)

# Split the data into training and testing datasets
training_dataset, test_dataset = dp.create_test_train_data(sdf_shapes, total_velocities, test_train_split)

# Initialise the model, loss function, and optimizer
model = CNN32_Model.CNN32_Model() if (model_type == 32) else CNN64_Model.CNN64_Model()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):  # loop over the dataset multiple times
    epoch_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(training_dataset, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        tensor_inputs = torch.FloatTensor(inputs).reshape(1, 1, inputs.shape[0], inputs.shape[0])
        tensor_labels = torch.FloatTensor(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + optimize
        outputs = model(tensor_inputs)
        loss = criterion(outputs, tensor_labels)
        loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += outputs.shape[0] * loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    print(epoch+1, epoch_loss / len(training_dataset))
print('Finished Training')


# Test the model's accuracy
total_mse = 0
for data in test_dataset:
    inputs, labels = data
    tensor_inputs = torch.FloatTensor(inputs).reshape(1, 1, inputs.shape[0], inputs.shape[0])
    tensor_labels = torch.FloatTensor(labels)

    tensor_outputs = model(tensor_inputs)

    predicted = tensor_outputs[0][1].detach().numpy()
    actual = tensor_labels.detach().numpy()

    rmse = np.sqrt(np.mean((actual - predicted)**2))
    n_rmse = rmse / np.amax(actual)-np.amin(actual)
    total_mse += n_rmse

accuracy = (total_mse / len(test_dataset)) * 100
print(f"Error: {accuracy}%")

# Save the model to a file
torch.save(model, f"Models/{model_name}.pt")
