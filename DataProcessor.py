import os
import csv
import pandas as pd
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

class DataProcessor:    
    def read_data(self, directory):
        raw_data = []
        for file in sorted(os.listdir(directory)):
            filepath = os.path.join(directory, file)
            with open(filepath, newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
                raw_data.append(data)

        clean_data = []
        for d in raw_data:
            clean_datum = pd.DataFrame(d).iloc[1:-1, 1:-2]
            clean_data.append(clean_datum.transpose())

        return clean_data

    # Splits the data into training and testing datasets
    def create_test_train_data(self, shape_data, velocity_data, train_size):
        temp = []
        for i in range(len(shape_data)):
            input_example = shape_data[i].astype(float)
            output_example = velocity_data[i].values.astype(float)
            temp.append((input_example, output_example))

        data_tensor = torch.FloatTensor(temp)

        # Split training dataset into training, validation, and test
        training_size = int(train_size * len(data_tensor))
        test_size = len(data_tensor) - training_size

        training_dataset, test_dataset = torch.utils.data.random_split(data_tensor, [training_size, test_size])
        return training_dataset, test_dataset

    # Create the signed distance field for each shape
    def create_sdf_shapes(self, clean_shapes):
        sdf_shapes = []
        for s in clean_shapes:
            # Create a mask for the inside and outside of the shape
            shape = s.astype(float)
            inside_mask = shape == -1
            outside_mask = shape == 0

            # Compute the distance transform for both the inside and outside
            distance_inside = distance_transform_edt(inside_mask)
            distance_outside = distance_transform_edt(outside_mask)

            # Create the signed distance field
            signed_distance_field = distance_outside - distance_inside

            # Normalize the signed distance field
            max_distance = np.max(np.abs(signed_distance_field))
            if max_distance > 0:
                signed_distance_field = signed_distance_field / max_distance

            # Set the values inside the original shape to 0
            signed_distance_field[inside_mask] = 0
            sdf_shapes.append(signed_distance_field)

        return sdf_shapes

    # Calculate total velocities from horizontal and vertical components
    def calculate_total_velocities(self, horizontal, vertical):
        total_velocities = []
        for i in range(len(horizontal)):
            total_velocity = np.sqrt(np.square(vertical[i].astype(float)) + np.square(horizontal[i].astype(float)))
            total_velocities.append(total_velocity)
        return total_velocities
    