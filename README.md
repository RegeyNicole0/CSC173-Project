# CSC173 Project
 CSC173 Collaborative Final Project

 PLEASE USE VIRTUAL ENVIRONMENT!!

## Local Setup 

1. Clone the repository.
2. Setup virtual environment and activate.
2. Install local requirements,
    **pip install -r requirements.txt**
5. Run main script,
    **py run.py**

## Quit Execution
1. Press 'esc' key.


## Folder: Model Training
best.pt - contains the best weights
last.pt - contains the last weights

## Folder: Test Resources
Contains Images and Videos for testing.
To test run this script in command line:
yolo task=detect mode=predict model=<modelname>.pt show=True source= <source to test>

Note: Make sure to copy the pretrained model, as well as resources to the main folder to test!!