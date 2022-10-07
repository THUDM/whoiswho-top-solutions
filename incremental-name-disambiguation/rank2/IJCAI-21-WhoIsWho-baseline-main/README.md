# IJCAI-21-WhoIsWho-baseline
Task-1: Incremental Name Disambiguation Baseline

**!!! You may want to read the *feature+bert.pptx* first for brief review.!!!**

### Running step:

+ **Data process** 
  + pre_gen_data.py: pre-generate training and test data for later training.
    + Downloading the IDF files via https://pan.baidu.com/s/1g1w2m20V4WPj0YNGYyF8Tw  passwd: y2ws
    + Training data here is composed of training set provided by the competition and na-v2 version of WhoIsWho (https://www.aminer.cn/whoiswho).
+ **Feature-engineering (36-d feature)**
  + ml_main.py: Main function for feature-engineer.

+ **Bert-based semantic model (41-d feature)**
  + dl_main.py: Main function for Bert-based embedding model.

#### Evaluation Mode
Go to folder *evaluation/*
+ processCandidate.py: Process the raw valid competition files.

+ evalFeatureMain.py: Load trained xgboost model and predict paper assignment result.



### Results.

+ **Feature-engineering (36-d feature)**

  + Without processing NIL (threshold = 0.0). 

  + Processing NIL by setting the pre-defined threshold, that is, for each unassigned paper, if its top predicted author score >= threshold, we assign the paper to the top predicted author.

    | Thresholds | 0.0   | 0.5   | 0.6   | 0.7   | 0.8   | 0.9   |
    | ---------- | ----- | ----- | ----- | ----- | ----- | ----- |
    | Precision  | 70.54 | 85.10 | 86.05 | 87.10 | 87.97 | 88.59 |
    | Recall     | 93.94 | 93.35 | 92.97 | 92.58 | 91.62 | 88.85 |
    | F1         | 80.57 | 89.03 | 89.37 | 89.76 | 89.76 | 88.77 |

    

  + Processing NIL by additional classifier via extracting score distribution feature of candidate authors.  

    (Refer to ppt)



+ **Bert-based semantic model (41-d feature)**

