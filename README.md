# Quick Start
This is to just download the ToneIt app and utilize the example
1. Clone the repository
2. Download (the model)[https://drive.google.com/file/d/1IpBXq09vFqbJ2RRlDDlFCmHbU3WlnpCf/view?usp=sharing] (50epochs, 16000objects)
3. Run the main application via the GoogleCOLAB through the `MainApp.ipynb` file

# Extensive Start
This is to build the model from scratch yourself
1. Clone the repository
2. Open the `CreateModel.ipynb` file in a GoogleCOLAB environment
3. Upload the trainTone.txt file to the GoogleCOLAB environment
4. Run the `CreateModel.ipynb` file (you can change parameters based on the `HyperParameterTuning.ipynb` file)
5. Download the `ToneItPipeline.tar.gz` archive from the GoogleCOLAB environment
6. Proceed to QuickStart with your ToneItPipeline model

# Notes
- Model evalution metrics are found in `CreateModel.ipynb` file for the 50epochs 16000objects model
- More performance and evaluation metrics from hyperparameter tuning are in `HyperParameterTuning.ipynb` though these are trained on less epochs
- The function that calls upon the model created from the other two jupyter notebooks is in `MainApp.ipynb`
- `data/` contains the data processed via the .java code in the same directory from the original dataset used to train the models
