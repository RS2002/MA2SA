# Ride Sharing

## 1. Dataset

The dataset used in this study is derived from the [yellow taxi data in Manhattan](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). 

~~The processed data can be found in the `./data` directory.~~

Considering the copyright, we have removed the processed data. However, the data processing code is available in the `./data` directory. Please download the dataset from the link provided above and use our code to process it.

## 2. How to Run

### 2.1 Pre-train

Execute the following command in the `./pretrain` directory:

```shell
python train.py --bi_direction 
```

### 2.2 Fine-tune

Execute the following command in the `./finetune` directory:

```shell
python train.py --bi_direction --pretrain_model_path <path_to_trained_model_from_stage_1>
```
