# About
The project consists of a few jupyter notebooks, python files, and some C files. The main goal is to create and test CNNs for steady laminar flow velocity field predictions.

### Python Files
These contain the finalised version of the CNN architectures, training loop, and model loading and testing.
### Jupyter Notebooks
These were used during the development process and largely contain similar code found in the python files, with some aditional sections to help testing and debugging.
### C Files
The C folder contains a modified version of the code found [here](https://github.com/dorchard/navier.git). It is used to generate the datasets used for training the models. Instruction on how to build and run it can be found at the linked github page.

# Building and Running
### Generate dataset
After running the navier file it will output `horizontal`, `vertical`, `shape` folders which will contain the data files. 
### Train network
Edit the `config.yaml` file and set parameters as needed. Make sure `dataset_path` is set to the directory that contains the three dataset folders.
Then run:
```
python TrainNetwork.py --config config.yaml
```
After training completes, the model will be saved using the name specified in config file.
### Run model
To run the model, edit the config and make sure `model_path` is pointing at the correct model, and `shape_data_file` is pointing at the folder containg input images for the model.
Then run:
```
python ModelLoader.py --config config.yaml
```
After running, images containing model outputs will be saved in the directory specified in the config file.
