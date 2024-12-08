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
  
  | Dataset           | Variety                                                                 | Preprocessing Needed                                                                         | Rawness                                                                                     | Usability for Victim Detection                                           |
  |--------------------|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
  | C2A               | Variety of disaster images with artificial human silhouette pre-embedded | Image resizing will be needed to bring all the images to uniform dimension.                | Usable for training for human detection application                                       | Focused on humans, with annotation to validate the model output|
  | Custom Dataset made from AIDER + UAV-Human | AIDER provides five different disaster backgrounds. UAV-Human provides 11,805 human poses as viewed from UAV | Custom developed engine to randomly overlay humans on the background and to generate annotation | The custom-generated data has actual humans in the background, aiding better feature learning | Focused on actual humans in the disaster environment for better learning|


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
- Fine-tune C2A train dataset with custom neck using YOLOv8n
- Fine-tune C2A train dataset using YOLOv9c

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

- Step-1: Initial Preprocessing of C2A Dataset
  - Click (Single click or double click whatever works) on the `c2a_dataset_image_preprocessing.ipynb` file.
  - You will notice the file opened.
  - Update the `images_base_path` variable with proper C2A Dataset path.
  - Click `Run` button from the Menu bar and select the option of your interest (`Run Cell` or `Run All` button).
  - You can look at the execution results within the file and interpret accordingly.
  - Now come back to the `Pneumonia_Detection_Preprocessing.ipynb` file and proceed with the image padding section which is the last part of this file execution.

- Step-2: Denoise the Images in C2A Dataset
  - Click (Single click or double click whatever works) on the `C2A_Dataset_Images_Denoising.ipynb` file.
  - You will notice the file opened.
  - Make sure you mount your Google Drive account.
  - Update the `input_folder` variable with proper C2A Dataset train images path and `output_folder` with proper path to save the denoised images.
  - Click `Run` button from the Menu bar and select the option of your interest (`Run Cell` or `Run All` button).
  - You can look at the execution results within the file and interpret accordingly.

- Step-3: Denoise the Images in the Created Synthetic Dataset
  - Click (Single click or double click whatever works) on the `Denoising_Custom_Dataset_Images.ipynb` file.
  - You will notice the file opened.
  - Make sure you mount your Google Drive account.
  - Update the `input_folder` variable with proper C2A Dataset train images path and `output_folder` with proper path to save the denoised images.
  - Click `Run` button from the Menu bar and select the option of your interest (`Run Cell` or `Run All` button).
  - You can look at the execution results within the file and interpret accordingly.

## Model Training and Results Validation 
- Be it any file, ensure you mount your Google Drive account in Google Colab file whatever you execute.

- Step-1: Create a Synthetic Dataset using AIDER Dataset and HUMAN Images.
  - Upload the AIDER and HUMAN images datasets to your Google Drive. You can take [this](https://drive.google.com/file/d/1im8K5qkV-8u-LqruvuFQ7NSQaoZbRScN/view?usp=sharing) and [this](https://drive.google.com/file/d/1mr0GUhlVtoW_WMkEEG_wkopTF6QJXtKx/view?usp=sharing) as reference.
  - Open the [Custom_AIDER_and_Human_Dataset.ipynb](https://github.com/MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs/blob/main/Generated_Custom_AIDER_and_Human_Dataset/Custom_AIDER_and_Human_Dataset.ipynb) file into Google Colab. 
    - In `!unzip <</content/drive/MyDrive/human.zip>> -d /content/human` and ` !unzip <</content/drive/MyDrive/AIDER.zip>> -d /content/AIDER` steps, give the correct path of your HUMAN images and AIDER images. Don't change the destinations.
    - Except the last cell of the file, execute the entire file after updating the above items.
    - Now, update `drive_output_dir` with proper Google Drive path of your interest in the last cell of the file, and execute this as well.
    - Wait for sometime after executing these steps. Then go inside the path in your Google Drive and ensure your dataset is visible in it.

- Step-2: Download the C2A Dataset from Kaggle and Unzip it. Refer [Download_Dataset_from_Kaggle_and_Unzip.ipynb](https://github.com/MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs/blob/main/Download_Datasets_and_Unzip/Download_Dataset_from_Kaggle_and_Unzip.ipynb) for better understanding.
  - Create a folder at the path of your interest in your Google Drive.
  - Open a Google Colab Notebook and execute the following steps to do the same:
    ```
    from google.colab import drive
    drive.mount('/content/drive')
      
    !wget "https://www.kaggle.com/api/v1/datasets/download/rgbnihal/c2a-dataset" -O "{give your Google Drive folder path here}/C2A_Dataset.zip"

    !unzip "{give your Google Drive folder path here}/C2A_Dataset.zip" -d "{give your Google Drive folder path here}/C2A_Dataset"
    ```
  - Wait for sometime after executing these steps. Then go inside the path in your Google Drive and ensure your dataset is visible in it Refer [this](https://drive.google.com/drive/folders/1BYrmnt_NF8o8sGeItjliYq8fl1GYeMk8?usp=sharing) to understand how the created dataset structure looks like.

- Step-3: Fine-tune YOLOv7 models using the C2A and created Synthetic Datasets. 

  ***This will explain you how to Fine-tune the YOLOv7 model using the C2A train dataset and again fine-tuning the resultant model with the created synthetic train dataset.***
  
  - Ensure step-1 is successful before executing this.   
  - Take the [Yolov7_C2A_Custom.ipynb](https://github.com/MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs/blob/main/C2A_and_CustomNewData_Yolo_v7/Yolov7_C2A_Custom.ipynb) file and open in your Google Colab.
  - Take the [data.yaml](https://github.com/MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs/blob/main/C2A_and_CustomNewData_Yolo_v7/data.yaml) and [data_C2A.yaml](https://github.com/MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs/blob/main/C2A_and_CustomNewData_Yolo_v7/data_C2A.yaml) files into your local system.
  - Update the following values in `data.yaml` file:
    - ```
      # Paths to dataset directories
      train: <</content/drive/MyDrive/YOLOv_Dataset/train/images>>
      val: <</content/drive/MyDrive/YOLOv_Dataset/val/images>>
      test: <</content/drive/MyDrive/YOLOv_Dataset/test/images >>
      ```
  - Update the following values in `data_C2A.yaml` file:
    - ```
      # Paths to dataset directories
      val: <</content/drive/MyDrive/YOLOv_Dataset/val/images>>
      train: <</content/drive/MyDrive/C2A_Dataset/C2A_Dataset/new_dataset3/train/images>>
      test: <</content/drive/MyDrive/C2A_Dataset/C2A_Dataset/new_dataset3/test/images>>
      ```
  - Then upload the updated `data.yaml` ([refer this to see how it looks](https://drive.google.com/file/d/157zaKKIgzrY3kp19n-9ZS9TDk5bVjQaY/view)) and `data_C2A.yaml` ([refer this to see how it looks](https://drive.google.com/file/d/1-A6crux1yo7ugq8_BABNC5zrUJFlGaSO/view?usp=sharing)) files to your Google Drive.
  - Then, update the proper Google Drive paths of `data.yaml` and `data_C2A.yaml` files in all the train and test segments of your colab notebook. Basically, you need to update the following code lines.
    ```
    !python train.py --weights yolov7.pt --data "<</content/drive/MyDrive/C2A_Dataset/data_C2A.yaml>>" --workers 8 --batch-size 8 --img 416 --cfg cfg/training/yolov7.yaml --name yolov7 --epochs 2 --hyp data/hyp.scratch.p5.yaml
    !python test.py --data <</content/drive/MyDrive/C2A_Dataset/data_C2A.yaml>> --img 416 --batch 8 --conf 0.35 --iou 0.65 --device 0 --weights /content/yolov7/runs/train/yolov7/weights/best.pt --name yolov7_416_val
    !python detect.py --weights <</content/yolov7/runs/train/yolov7/weights/best.pt>> --source "<<your path to created synthetic datatse>>/test/images"
    !cp -r /content/yolov7/runs/train/yolov7 /content/drive/MyDrive/<<your path to save results>>
    
    !python train.py --weights /content/yolov7/runs/train/yolov7/weights/best.pt --data "<</content/drive/MyDrive/C2A_Dataset/data.yaml>>" --workers 8 --batch-size 8 --img 416 --cfg cfg/training/yolov7.yaml --name yolov7 --epochs 2 --hyp data/hyp.scratch.p5.yaml
    !python test.py --data <</content/drive/MyDrive/C2A_Dataset/data.yaml>> --img 416 --batch 8 --conf 0.35 --iou 0.65 --device 0 --weights /content/yolov7/runs/train/yolov72/weights/best.pt --name yolov7_416_val
    !python detect.py --weights /content/yolov7/runs/train/yolov72/weights/best.pt --source "<<your path to created synthetic datatse>>/test/images"
    !cp -r /content/yolov7/runs /content/drive/MyDrive/<<your path to save results>>/
    ```
  - Execute all the steps of the notebook file.
  - You can refer to the following paths to view the executed results. Accordingly, you can view the results obtained for your execution as well. Take [this](https://drive.google.com/drive/folders/1VkneXThQ74QsMTOCyzmeQaDDnpStbLtN?usp=sharing) as reference.
    - All the train results will be usually available inside `train/yolov7` and `train/yolov72` paths of the saved results folder. Take [this](https://drive.google.com/drive/folders/1VkneXThQ74QsMTOCyzmeQaDDnpStbLtN?usp=sharing) as reference.
    - Following are the results you are able to view. Each of them are available in file.
      - `results.png`
      - `P_curve.png`
      - `R_curve.png`
      - `PR_curve.png`
    - Model weights are available inside `train/yolov7/weights` and `train/yolov72/weights`. Accordingly, you can view the results obtained for your execution as well. Take [this](https://drive.google.com/drive/folders/1VkneXThQ74QsMTOCyzmeQaDDnpStbLtN?usp=sharing) as reference.

- Step-4: Fine-tune YOLOv7 using the C2A train dataset.
  - Take the [c2a-using-yolov7-base.ipynb](https://github.com/MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs/blob/main/C2A_training_using_yolov7_base/c2a-using-yolov7-base.ipynb) file and open it in your **Kaggle account**.
  - At the right-side of your Kaggle notebook, you will have an option to select the dataset. Click the corresponding dropdown and load the C2A dataset from there.
  - Take the [data_c2a_using_yolov7.yaml](https://github.com/MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs/blob/main/C2A_training_using_yolov7_base/data_c2a_using_yolov7.yaml) file to your local system.
  - Update the following values in this file:
    - ```
      # Paths to dataset directories
      val: <</kaggle/working/c2a-dataset/C2A_Dataset/new_dataset3/val/images>>
      train: <</kaggle/working/c2a-dataset/C2A_Dataset/new_dataset3/train/images>>
      test: <</kaggle/working/c2a-dataset/C2A_Dataset/new_dataset3/test/images>>
      ```
  - Then upload the updated `data_c2a_using_yolov7.yaml` file to your Kaggle notebook (in `/kaggle/working` path).
  - Execute all the steps of the Kaggle notebook file.
  - After executing `Download all the results from Kaggle to Local machine` section, you will find the same zip file at the right-side of your Kaggle notebook (in `/kaggle/working` path).
    - Click on the 3 vertical dots of that file and you will get an option `Download` to download the file.
  - Download the zip file to your local system, unzip it and then upload it back to your Google Drive. Take [this](https://drive.google.com/drive/folders/1FutSReRKyaPrfKTQnbddUUkcQHcdVB2w?usp=sharing) as reference to view the same.
  - You can refer to the following paths to view the executed results. Accordingly, you can view the results obtained for your execution as well.
    - All the train results will be usually available [here](https://drive.google.com/drive/folders/1Nzytew_5e8VNMP7ZVEmo5K4-4vq5ZMXs?usp=sharing). Following are the results you are able to view. Each of them are available in file.
      - `results.png`
      - `P_curve.png`
      - `R_curve.png`
      - `PR_curve.png`
    - Model weights are available [here](https://drive.google.com/drive/folders/1zkki2EVikCp33SyTZtwKavx1kobjrkYr?usp=sharing).
    - After validating the fine-tuned images while executing the code, all the detected humans related images will be visible [here](https://drive.google.com/drive/folders/1ywi_hcMYIn1Nf2PLVHpAYsnkT-dMPr-O?usp=sharing).

- Step-5: Fine-tune YOLOv9c using the C2A train dataset.
  - Take the [c2a_using_yolov9c_base.ipynb](https://github.com/MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs/blob/main/C2A_training_using_yolov9c_base/c2a_using_yolov9c_base.ipynb) file and open it in your Google Colab.
  - At the right-side of your Kaggle notebook, you will have an option to select the dataset. Click the corresponding dropdown and load the C2A dataset from there.
  - Take the [data.yaml](https://github.com/MNiazM/Human-Presence-Detection-in-Aerial-Distress-Scenario-Photographs/blob/main/C2A_training_using_yolov9c_base/data.yaml) file to your local system.
  - Update the following values in this file:
    - ```
      # Paths to dataset directories
      val: <</content/drive/MyDrive/MS_in_AAI/Courses_and_Projects/AAI-521_CV/AAI-521_Final_Team_Project/CV_Datasets/C2A_Dataset/C2A_Dataset/new_dataset3/train/images>>
      train: <</content/drive/MyDrive/MS_in_AAI/Courses_and_Projects/AAI-521_CV/AAI-521_Final_Team_Project/CV_Datasets/C2A_Dataset/C2A_Dataset/new_dataset3/val/images>>
      test: <</content/drive/MyDrive/MS_in_AAI/Courses_and_Projects/AAI-521_CV/AAI-521_Final_Team_Project/CV_Datasets/C2A_Dataset/C2A_Dataset/new_dataset3/test/images>>
      ```
  - Then upload the updated `data.yaml` file to your Google Drive path. Refer [this](https://drive.google.com/drive/folders/1b4uqICajs1X_zluMQdnsk556X0z3b4w4?usp=sharing) to understand where/how you can upload the file.
  - In the `Define base paths for dataset images and labels. Check the paths existence` section of your file, update the `base_path` variable with the full path of C2A dataset.
  - Update the `data_yaml_path` in the `Ensure data.yaml is existing` section of your file.
  - Update the following lines in `Transfer the results to Google Drive` section of your file.
    ```
    check_paths_existence("<</content/drive/MyDrive/MS_in_AAI/Courses_and_Projects/AAI-521_CV/AAI-521_Final_Team_Project/>>")
    destination_folder = "<</content/drive/MyDrive/MS_in_AAI/Courses_and_Projects/AAI-521_CV/AAI-521_Final_Team_Project/yolov9c_base_runs>>"
    copied_yolov9c_results_path = "<</content/drive/MyDrive/MS_in_AAI/Courses_and_Projects/AAI-521_CV/AAI-521_Final_Team_Project/yolov9c_base_runs>>"
    ```
  - Update the following lines in `Copy base models as well to Google Drive for safer side` section of your file.
    ```
    base_model_source = "/content/yolov9c.pt"
    base_model_destination = "/content/drive/MyDrive/MS_in_AAI/Courses_and_Projects/AAI-521_CV/AAI-521_Final_Team_Project/yolov9c_base_model.pt"
    ```
  - 
  - You can refer to the following paths to view the executed results. Accordingly, you can view the results obtained for your execution as well.
    - All the train results will be usually available [here](https://drive.google.com/drive/folders/1-1VMTzoOKOjN5ZyeNAw6NjBMmIKs8uUm?usp=sharing). Following are the results you are able to view. Each of them are available in file.
      - `results.png`
      - `P_curve.png`
      - `R_curve.png`
      - `PR_curve.png`
    - Model weights are available [here](https://drive.google.com/drive/folders/10WeZwjbXjZAlHAO-_8Jqx5SExlUPSKcA?usp=sharing).
    - After validating the fine-tuned images while executing the code, all the detected humans related images will be visible [here](https://drive.google.com/drive/folders/1-1VMTzoOKOjN5ZyeNAw6NjBMmIKs8uUm?usp=sharing).

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
