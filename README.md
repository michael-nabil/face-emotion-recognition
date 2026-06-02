# Face-emotion-recognition
- Real-time application that detects faces and classifies their emotions.
  - Detecting faces using OpenCV's [haarcascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) light-weight face detector.
  - Classifing faces using finetuning MobileNetV2 model.
- Using WebSockets for realtime continuous video-data transfer between frontend and backend.

## 1. Experiments
Experiments and artifacts of finetuning MobileNetV2 are logged using MLFlow and DagsHub, check it out [here](https://dagshub.com/michael-nabil/face-emotion-recognition.mlflow/#/experiments/0/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D).

## 2. Face Emotion Classifier Model
- Using `MobileNetV2` pretrained CNN models.
- Finetunning on `two` phases by unfreezing layers gradually.
- logging Training and Validation loss and accuracy.
- Saving best model with least validation Loss.

## 3. Tech Stack
- Model development using TensorFlow.
- Backend API using python's FastApi.
- MLFlow and DagsHub for experiments and artifacts logging.
- Image preprocessing and haarcascade face detection using OpenCV.
- Matplotlib and Seaborn for plotting and visualization.
- Model trained on Kaggle in a well-structured Notebook, check it out [here](https://www.kaggle.com/code/michaelnabil88/face-expression-classification-ipynb).
