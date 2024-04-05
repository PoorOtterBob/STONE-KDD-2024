# Submission 126: STONE: A Spatio-temporal OOD Learning Framework Kills Both Spatial and Temporal Shifts

## 1. Introduction about the datasets
In our experiments, we used SD and GBA datasets which are generated from CA dataset, followed by [LargeST](https://github.com/liuxu77/LargeST/blob/main). For example, you can download CA dataset from the provided [link](https://www.kaggle.com/datasets/liuxu77/largest) and please place the downloaded archive.zip file in the `data/ca` folder and unzip the file. 
### Generating the SD and GBA sub-datasets from CA dataset
First of all, you should go through a jupyter notebook `process_ca_his.ipynb` in the folder `data/ca` to process and generate a cleaned version of the flow data. Then, please go through all the cells in the provided jupyter notebooks `generate_sd_dataset.ipynb` in the folder `data/sd` and `generate_gla_dataset.ipynb` in the folder `data/gla` respectively. Finally use the commands below to generate traffic flow data for our experiments. 
```
python generate_data_for_training.py --dataset sd --years 2019_2020_2021
python generate_data_for_training.py --dataset gba --years 2019_2020_2021
```
## 2. Data Processing

## 3. Environmental Requirments

## 4. Model Running
To run STONE, for example, you may execute this command in the terminal:
```
bash experiments/stone/run.sh
```
or directly execute the Python file in the terminal:
```
python experiments/stone/main.py --device cuda:0 --dataset SD --years 2019 --model_name stone --seed 0 --bs 64
```
