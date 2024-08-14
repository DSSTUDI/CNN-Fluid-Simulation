import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import DataProcessor
import sys
import CNN32_Model

# initialise variables
epochs = 50
test_train_split = 0.8
kernel_size = 3
learning_rate = 0.001
model_name = "SimpleNetwork32"


if len(sys.argv) > 1:
    epochs = int(sys.argv[1])
    test_train_split = float(sys.argv[2])
    kernel_size = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    model_name = sys.argv[5]
else:
    print("Using default values")

dp = DataProcessor.DataProcessor()

# Load the data
sdf_shapes = dp.create_sdf_shapes(dp.read_data('Dataset_32/shape'))
clean_horizontals = dp.read_data('Dataset_32/horizontal')
clean_verticals = dp.read_data('Dataset_32/vertical')
total_velocities = dp.calculate_total_velocities(clean_horizontals, clean_verticals)

# Split the data into training and testing datasets
training_dataset, test_dataset = dp.create_test_train_data(sdf_shapes, total_velocities, test_train_split)

# Initialise the model, loss function, and optimizer
model = CNN32_Model.CNN32_Model()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):  # loop over the dataset multiple times
    epoch_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(training_dataset, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        tensor_inputs = torch.FloatTensor(inputs).reshape(1, 1, 32, 32)
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
    tensor_inputs = torch.FloatTensor(inputs).reshape(1, 1, 32, 32)
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
