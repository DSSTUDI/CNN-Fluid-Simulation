import configargparse

# Initialize ConfigArgParse parser
p = configargparse.ArgParser()

# Define command-line arguments and configuration file options
p.add('-c', '--config', is_config_file=True, help='Path to config file')

p.add('-m', '--model_type', type=int, required=True, help='32x32 or 64x64 model to be used')

p.add('-k', '--kernel_size', type=int, help='Kernel size', default=3)
p.add('-s', '--stride', type=int, help='Stride length', default=1)
p.add('-p', '--padding', type=int, help='Padding amount', default=1)

p.add('-e', '--epochs', type=int, help='Epoch amount', default=50)
p.add('-tts', '--test_train_split', type=float, help='Test/Train data split ratio', default=0.8)
p.add('-lr', '--learning_rate', type=float, help='Learning rate of network', default=0.001)
p.add('-tds', '--training_data_size', type=int, help='Size of training data', default=1000)

p.add('-dd', '--dataset_path', required=True, help='Path to training data directory')

p.add('-n', '--model_name', help='Name for model to be saved', default='DefaultModel')

p.add('-mp', '--model_path', required=True, help='Path to model file')
p.add('-dp', '--shape_data_path', required=True, help='Path to shape data files')

p.add('-ex', '--examples', type=int, help='Amount of example images to create', default=10)
p.add('-ep', '--example_path', help='Path to save example images', default='Examples')



# Parse arguments
config = p.parse_args()
