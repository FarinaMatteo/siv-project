# Foreground vs Background Segmentation
  

## Introduction  
This repository will contain code and documentation concerning the final project for the course *Signal, Image & Video* taught at **University of Trento**.  

The aim of the project is to develop an "video-conferencing-like" algorithm to segment background and foreground, in order to run an image composition technique for background substitution. 

[ ADD EXAMPLES ]

## Installation  
The source code of the project is written in **Python 3.7**.  
Different versions of Python may eventually do just fine, but this cannot be guaranteed.   
It is kindly suggested that you setup a python virtual environment in order to properly isolate and install the required dependencies. To this aim, we suggest using **Anaconda**. 

### Anaconda Environment  
In order to setup an Anaconda environment with this project dependencies, run the following:  
```
$ conda create --name <your-env-name> python=3.7
$ conda activate <your-env-name>
(<your-env-name>) $ pip install -r requirements.txt  
```  

## What's inside this repo?  
Different proposed techniques are provided within different python files. 
- The default implementation inside __main.py__  uses gray scale frame differencing in order to segment the foreground and the background;  
- An improved implementation can be found in __hsv.py__ , where the HSV colorspace is leveraged in order to take into account the contribution of both brightness and saturation to segment the background and the foreground;  
- A customizable implementation in __trackbars.py__ lets users interactively configure proper values for the thresholds related to the saturation and the brightness masks.  
- A latter script allows to use the Optical Flow theory in order to keep track of the foreground mask over time. This can be found in __hsv_optflow.py__.

## Run the code  
[INSERT FURTHER EXPLANATIONS HERE]  
