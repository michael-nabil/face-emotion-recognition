# Face-emotion-recognition
>Real-time application that detects the face emotions

## 1. Experiments
Experiments and artifacts are logged using MLFlow and DagsHub and can be viewed from this [link](https://dagshub.com/michael-nabil/face-emotion-recognition.mlflow/#/experiments/0/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)

## 2. Methodology
- Using `MobileNetV2` pretrained CNN models.
- Finetunning on `two` phases by unfreezing layers gradually.
- logging Training and Validation loss and accuracy.
- Saving best model with least validation Loss.
