## Contents

1.  [Repository Name](#repository-name)
2.  [Title of the Project](#title-of-the-project)
3.  [Short Description and Objectives of the Project](#short-description-and-objectives-of-the-project)
4.  [Details about the Datasets](#details-about-the-datasets)
5.  [Goal of this Project](#goal-of-this-project)
6.  [Requirements of this Project](#requirements-of-this-project)
7.  [Algorithms which can be used](#algorithms-which-can-be-used)
8.  [Issues to focus on](#issues-to-focus-on)
9.  [Project Requirements](#project-requirements)
10. [Usage Instructions in Local System and Google Colab](#usage-instructions-in-local-system-and-google-colab)
11. [Authors](#authors)
12. [References](#references)
----------------------------------------------

# Repository Name
Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs

----------------------------------------------

# Title of the Project
Human presence detection in Aerial Distress scenario photographs

----------------------------------------------

# Short Description and Objectives of the Project
The effectiveness of search and rescue missions from disaster scenarios such as earthquakes, floods, building collapses and fire accident hinges on the rapid identification of victims among debris and rubble. For large-scale areas affected, aerial surveillance with computer vision tools can play a pivotal role in detecting victims, facilitating prompt rescue operations and delivering medical aid. This project focuses on developing a state-of-the-art computer vision model capable of identifying human presence from aerial distress images, thereby improving the speed and accuracy of disaster response efforts.

----------------------------------------------

# Details about the Datasets
- **Primary Fine-Tuning Dataset:**
  - ***C2A Dataset: Human Detection in Disaster Scenario:***
  - ***Description of C2A Dataset:***
    - **Number of images:** Train (6129), Test (2043) and Validation (2043)
    - **Dimension of images:** Ranging from 123x152 to 5184x3456
    - **Size of dataset:** 5.04 GB
  - ***Data Source:*** https://www.kaggle.com/datasets/rgbnihal/c2a-dataset

- **Secondary Fine-Tuning Dataset:**
  1. ***Combined Image set + Annotation from AIDER+UAV-Human Image sets***
    - **Backgrounds:** AIDER (Aerial Image Dataset for Emergency Response)
    - ***Description of AIDER Dataset:***
      - **Number of images:** 500 images per disaster class x 5 disaster class
      - **Dimension of images:** 1920 x 1080
      - **Size of dataset:** 275.7 MB
  - ***Data Source:*** https://zenodo.org/records/3888300
  
  2. ***Human Overlay: UAV-Human: Human behavior understanding with UAV***
     - ***Description of UAV-Human Dataset:***
       - **Number of images:** 11,805 images 
       - **Dimension of images:** Varying range with average of 400 x 500 
       - **Size of dataset:** 355 MB
     - ***Data Source:*** https://github.com/sutdcv/UAV-Human (bounding_box_train folder)

----------------------------------------------

# Goal of this Project
- **Goal:** This project focuses on developing a state-of-the-art computer vision model capable of identifying human presence from aerial distress images, thereby improving the speed and accuracy of disaster response efforts.
- **Significance:** The effectiveness of search and rescue missions from disaster scenarios such as earthquakes, floods, building collapses and fire accident hinges on the rapid identification of victims among debris and rubble. For large-scale areas affected, aerial surveillance with computer vision tools can play a pivotal role in detecting victims, facilitating prompt rescue operations and delivering medical aid.

----------------------------------------------

# Requirements of this Project
- **Considered Data Mesaures:** The following image depicts the measures we took to achieve the goal of this project.
  ![img.png](img.png)
- We followed 2 stages of finetuning.
  - C2A is used for initial fine tuning of the model and pruning the model architecture as it is a light-weight dataset with human centric focus.
  - Custom dataset from (AIDER+UAV-Human overlay) has more realistic human images and annotation over disaster background. Hence, this will be used for second level fine tuning.
- Feature extraction plan:
  - YoloV8, with Feature Pyramid Network (FPN) with image adaptation layer and fine tuning with above datasets phase wise is the planned approach.
  - We are also trying with YoloV9c, by customizing the backbone using various architectures to understand the performance of the human detection.
  - The methodology will be tuned further based on intermediate results.

----------------------------------------------

# Algorithms which can be used
- Fine-tune C2A train dataset using YOLOv7
- Finetune C2A train dataset with custom neck using YOLOv8n
- Finetune C2A train dataset using YOLOv9c

----------------------------------------------

# Issues to focus on
- Create a custom dataset using AIDER Dataset and HUMAN Images.
- Fine-tune YOLOv7 using the C2A train dataset.
  - Validate the obtained fine-tuned model using created custom validation dataset.
- Fine-tune the resultant model (after training with YOLOv7 on C2A train dataset) with the created custom train dataset.
  - Validate the obtained fine-tuned model using created custom validation dataset
- Fine-tune YOLOv7 using the C2A train dataset.
  - Validate the obtained fine-tuned model using C2A test/validation dataset.
- Fine-tune YOLOv9c using the C2A train dataset.
  - Validate the obtained fine-tuned model using C2A test/validation dataset.

----------------------------------------------

# Project Requirements
- ultralytics
- ipywidgets
- opencv-python
- tensorflow
- torch
- torchvision
- pandas
- numpy
- jupyter
- notebook
- tqdm
- joblib
- scipy
- scikit-image
- scikit-learn
- starlette
- seaborn

----------------------------------------------

# Usage Instructions in Local System and Google Colab
- Clone using HTTPS
```commandline
https://github.com/MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs.git
```
OR - 

- Clone using SSH
```commandline
git@github.com:MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs.git
```

OR -

- Clone using GitHub CLI
```commandline
gh repo clone MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs
```
 
- Switch inside the Project Directory
```commandline
cd Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs
```

- Install Requirements
```commandline
pip3 install -r requirements.txt
```

- Switch inside the Dataset preprocessing path
```commandline
cd Datasets_Image_Preprocessing
```

- Open your terminal (Command Prompt in Windows OR Terminal in MacBook)
- Type any one of the below commands based on the software installed in your local system. You will notice a frontend webpage opened in the browser.
```commandline
jupyter notebook
```
OR -
```commandline
jupyter lab
```

## Dataset Preprocessing

- Step-1:
  - Click (Single click or double click whatever works) on the `c2a_dataset_image_preprocessing.ipynb` file.
  - You will notice the file opened.
  - Update the `images_base_path` variable with proper C2A Dataset path.
  - Click `Run` button from the Menu bar and select the option of your interest (`Run Cell` or `Run All` button).
  - You can look at the execution results within the file and interpret accordingly.
  - Now come back to the `Pneumonia_Detection_Preprocessing.ipynb` file and proceed with the image padding section which is the last part of this file execution.

- Step-2:
  - Click (Single click or double click whatever works) on the `C2A_Dataset_Images_Denoising.ipynb` file.
  - You will notice the file opened.
  - Make sure you mount your Google Drive account.
  - Update the `input_folder` variable with proper C2A Dataset train images path and `output_folder` with proper path to save the denoised images.
  - Click `Run` button from the Menu bar and select the option of your interest (`Run Cell` or `Run All` button).
  - You can look at the execution results within the file and interpret accordingly.

- Step-3:
  - Click (Single click or double click whatever works) on the `Denoising_Custom_Dataset_Images.ipynb` file.
  - You will notice the file opened.
  - Make sure you mount your Google Drive account.
  - Update the `input_folder` variable with proper C2A Dataset train images path and `output_folder` with proper path to save the denoised images.
  - Click `Run` button from the Menu bar and select the option of your interest (`Run Cell` or `Run All` button).
  - You can look at the execution results within the file and interpret accordingly.

## Model Training

- Create a custom dataset using AIDER Dataset and HUMAN Images.
- Fine-tune YOLOv7 using the C2A train dataset.
  - Validate the obtained fine-tuned model using created custom validation dataset.
- Fine-tune the resultant model (after training with YOLOv7 on C2A train dataset) with the created custom train dataset.
  - Validate the obtained fine-tuned model using created custom validation dataset
- Fine-tune YOLOv7 using the C2A train dataset.
  - Validate the obtained fine-tuned model using C2A test/validation dataset.
- Fine-tune YOLOv9c using the C2A train dataset.
  - Validate the obtained fine-tuned model using C2A test/validation dataset.

----------------------------------------------

# Authors
| Author            | Contact Details       |
|-------------------|-----------------------|
| Mohamed Niaz M    | mniazm@sandiego.edu   |
| Ravi Teja Kothuru | rkothuru@sandiego.edu |
| Shruthi AK        | sak@sandiego.edu      |
----------------------------------------------

# References

1. Nihal, R. A., Yen, B., Itoyama, K., & Nakadai, K. (2024). UAV-Enhanced Combination to Application: Comprehensive Analysis and Benchmarking of a Human Detection  Dataset for Disaster Scenarios. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2408.04922
2. Kyrkou, C., & Theocharides, T. (2019). Deep-Learning-Based aerial image classification for emergency response applications using unmanned aerial vehicles. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 517–525. https://doi.org/10.1109/cvprw.2019.00077
3. Li, T., Liu, J., Zhang, W., Ni, Y., Wang, W., & Li, Z. (2021). UAV-Human: A Large Benchmark for Human Behavior Understanding with Unmanned Aerial Vehicles. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2104.00946
