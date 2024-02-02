# VlogQA: Task, Dataset, and Baseline Models for Vietnamese Spoken-Based Machine Reading Comprehension
This project presents VlogQA - a new MRC dataset for Spoken-based language in Vietnamese.     
The VlogQA consists of 10,076 question-answer pairs based on 1,230 transcript documents. 
Please note that the dataset is available for research purposes only. 

## Data Inquiries
Please contact Mr. Son Luu (Email: sonlt@uit.edu.vn) for the dataset.  
Alternative email: son.lt1103@gmail.com    

## How to run the code  
For training: please run the train.py script.   
For evaluation: please run the test.py script.

Required params (for both train.py and test.py):  
--path: Path to the VlogQA dataset  
--type: Type of model (eg. bert, xlm-r, phobert, etc). Default is auto   
--model: the path to pre-trained model or slug as shown in the Hugging Face website (eg. xlm-r-base)   
--output_path: Path to the output directory   
--is_test: Pass this param in the test.py if you want to run the evaluation on the test set. If none, the code will run evaluation on the development (dev) set.   

## Publication 
tba 
