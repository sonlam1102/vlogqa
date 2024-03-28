# VlogQA: Task, Dataset, and Baseline Models for Vietnamese Spoken-Based Machine Reading Comprehension
This project presents VlogQA - a new MRC dataset for Spoken-based language in Vietnamese.     
The VlogQA consists of 10,076 question-answer pairs based on 1,230 transcript documents. 
Please note that the dataset is available for research purposes only. 

## Data Inquiry
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
Thinh Ngo, Khoa Dang, Son Luu, Kiet Nguyen, and Ngan Nguyen. 2024. VlogQA: Task, Dataset, and Baseline Models for Vietnamese Spoken-Based Machine Reading Comprehension. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1310–1324, St. Julian’s, Malta. Association for Computational Linguistics.

Link to publication: https://aclanthology.org/2024.eacl-long.79/  
## Citation 
```
@inproceedings{ngo-etal-2024-vlogqa,
    title = "{V}log{QA}: Task, Dataset, and Baseline Models for {V}ietnamese Spoken-Based Machine Reading Comprehension",
    author = "Ngo, Thinh  and
      Dang, Khoa  and
      Luu, Son  and
      Nguyen, Kiet  and
      Nguyen, Ngan",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.79",
    pages = "1310--1324",
    abstract = "This paper presents the development process of a Vietnamese spoken language corpus for machine reading comprehension (MRC) tasks and provides insights into the challenges and opportunities associated with using real-world data for machine reading comprehension tasks. The existing MRC corpora in Vietnamese mainly focus on formal written documents such as Wikipedia articles, online newspapers, or textbooks. In contrast, the VlogQA consists of 10,076 question-answer pairs based on 1,230 transcript documents sourced from YouTube {--} an extensive source of user-uploaded content, covering the topics of food and travel. By capturing the spoken language of native Vietnamese speakers in natural settings, an obscure corner overlooked in Vietnamese research, the corpus provides a valuable resource for future research in reading comprehension tasks for the Vietnamese language. Regarding performance evaluation, our deep-learning models achieved the highest F1 score of 75.34{\%} on the test set, indicating significant progress in machine reading comprehension for Vietnamese spoken language data. In terms of EM, the highest score we accomplished is 53.97{\%}, which reflects the challenge in processing spoken-based content and highlights the need for further improvement.",
}

```
