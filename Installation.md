## Overall environment
Ubuntu: 20.04.3   

PyTorch: 1.10.2  

Cuda: 11.3.0   

Python: 3.8

## Step 1: Download the project 

    git clone https://github.com/Hong-yu-Zhang/GAB-Net

## Step 2: Packages installation
The GAB-Net is implemented under the framework of ultralytics, and the following dependencies are required:
```
    pip install ultralytics
    pip install thop
    pip install sentry_sdk
```
    
## Step 3: Move the project folder to sitepackages.
    rm -rf [Your Path]/python3.X/dist-packages/ultralytics/
    cp -r ./ultralytics/ultralytics /usr/local/lib/python3.8/dist-packages/
    

