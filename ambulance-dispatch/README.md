# Ambulance Dispatch

## 1. Data Source (New York City )

1. Hospital Location: [Directory of 220 Hospitals](https://profiles.health.ny.gov/directory/hospitals)
2. EMS Location: [Regional EMS Agencies - The Regional Emergency Medical Services Council of New York City, Inc.](https://nycremsco.org/regional-ems-agencies/)
3. Incident Dispatch: [EMS Incident Dispatch Data | NYC Open Data](https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj/about_data)
4. Zip Codes: [US Zip Codes Database | Simplemaps.com](https://simplemaps.com/data/us-zips)



## 2. How to Run

### 2.1 Pre-train

Execute the following command in the `./pretrain` directory:

```shell
python train.py
```

If you want to simulate a pandemic scenario, please add the `--compression` parameter.



### 2.2 Fine-tune

Execute the following command in the `./finetune` directory:

```shell
python train.py --pretrain_model_path <path_to_trained_model_from_pretraining>
```