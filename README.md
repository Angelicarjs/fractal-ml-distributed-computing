# fractal-ml-distributed-computing


## Workflow Overview

**Objective:** Use distributed computing to implement and evaluate a machine learning algorithm for **land cover classification** using the **FRACTAL dataset** (https://huggingface.co/datasets/IGNF/FRACTAL ).

**Tools Used:**  
- **Data storage:** S3  
- **Data preprocessing:** PySpark  
- **ML algorithm:** PySpark / MLlib  
- **Processing platform:** AWS EMR  

---

## 1. Data Access
- The dataset is stored in **S3**, already divided into **train, validation, and test** subsets.  
- Spark reads the data directly from S3 paths for distributed processing.

## 2. Data Preprocessing
- Check emphy or zero columns.
- Check for null values.  
- Check the balance of the classes.  
- **Normalize numeric columns**, including **height normalization** (e.g., subtract minimum *z* per patch or within a radius).

## 3. Model Implementation
- Train a **supervised classification model** (e.g., Random Forest or Gradient Boosted Trees) using PySpark MLlib.  

## 4. Model Evaluation
- Evaluate the model on **validation** and **test** subsets.  
- Compute metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.  
- Visualize confusion matrix and feature importance.
