# Computer Vision Project
This repo contains code for the final proejct COMS4713 at Columbia University. Our goal is to produce a sort of survey of the techniques we covered in class for image processing, and using them to derive useful information from the [OCELOT dataset](https://lunit-io.github.io/research/ocelot_dataset/), an open data set of histopathological images. At a minimum, we're aiming to get within a reasonable margin of error for the cell annotations (i.e identifying both the total number of cells, and to distinguish between tumorous and non-tumorous cells).


# Structure
The driving code is in process_ocelot_data.py. It's designed to apply a technique to a given subset of images from the dataset and produce the results. Processing methods are defined in image_processing_methods/, and are run on each image in the dataset. 


# Setup
You'll first need to clone the dataset into the root directory of the project. It can be found at https://zenodo.org/records/8417503. The code assumes that the dataset is in the root directory of the project, and is called 'ocelot_testing_data'. Once you've cloned the dataset, you can setup your python environment by running:

```bash
conda create -n ocelot python=3.10
conda activate ocelot
pip install -r requirements.txt
```

or using venv:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Running the Code
Again, the driving code is in process_ocelot_data.py. You can run it by running:
```bash
python process_ocelot_data.py <folder> <type> [--image-limit N] [--method METHOD]
```

where <folder> is one of 'test', 'train', or 'val', and <type> is one of 'cell' or 'tissue'. The --image-limit flag is optional and specifies the number of images to process. The --method flag is optional and specifies the method to use for processing the images.



# Current Techniques Used
1. `cell_binary`: uses Thresholding and the opencv function for identifying cells in the image.
2. `tissue_binary`: uses Thresholding and the opencv function for identifying tissue in the image.
