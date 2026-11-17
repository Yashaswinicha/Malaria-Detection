BIOCON MALARIA DETECTION - PYCHARM PROJECT
==========================================

PROJECT CONTENTS
----------------
malaria_utils.py  Shared dataset, CNN, device and prediction functions
train.py          Trains and evaluates the custom PyTorch CNN
predict.py        Predicts one image from the command line
check_device.py   Checks CPU / Apple MPS / NVIDIA CUDA support
app.py            Streamlit website
requirements.txt  Python dependencies

1. OPEN IN PYCHARM
------------------
Open the BIOCON_Malaria_PyCharm folder as a PyCharm project.

2. CREATE / SELECT A PYTHON ENVIRONMENT
---------------------------------------
In PyCharm, open Settings > Project > Python Interpreter and create/select
a project virtual environment.

3. INSTALL DEPENDENCIES
-----------------------
Open the PyCharm Terminal and run:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

On an Apple Silicon Mac, install the current PyTorch and torchvision builds
that support macOS and MPS. Do not install NVIDIA CUDA packages on a Mac.

4. ADD THE DATASET
------------------
Preferred folder layout:

    BIOCON_Malaria_PyCharm/
        cell_images/
            Uninfected/
                image1.png
                image2.png
            Parasitized/
                image1.png
                image2.png

Alternatively, put the dataset ZIP in the project root and name it:

    dataset.zip

The training script will attempt to extract it and locate the two class folders.
If your ZIP has a different layout or different class folder names, inspect it
and adjust the folders to match the layout above.

5. CHECK DEVICE
--------------
Run check_device.py in PyCharm.
On an M4 Mac, it should select Apple Metal (MPS) if supported; otherwise CPU.

6. TRAIN THE CNN
----------------
Run train.py in PyCharm.
The script creates:
    outputs/malaria_cnn_best.pth
    outputs/training_curves.png
    outputs/confusion_matrix.png
    outputs/metrics.json

Training can take time. Start with a smaller dataset subset or fewer epochs
if you need to test that the setup works before a full training run.

7. RUN THE WEBSITE
------------------
After training, run this command in the PyCharm Terminal:

    python -m streamlit run app.py

Streamlit will print a local URL, usually http://localhost:8501.

8. OPTIONAL: PREDICT ONE IMAGE FROM TERMINAL
---------------------------------------------
    python predict.py path/to/your/cell_image.png

IMPORTANT
---------
This is an educational prototype, not a clinically validated diagnostic device.
Do not use its predictions for medical decisions.

The code uses a random image-level split. If multiple images originate from the
same patient, slide, or source, a group-based split is preferable to avoid
overestimating generalisation.
