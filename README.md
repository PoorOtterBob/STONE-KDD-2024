# KDD 2024 Submission 126: STONE: A Spatio-temporal OOD Learning Framework Kills Both Spatial and Temporal Shifts
This is the official repository of our KDD 2024 submission 126 paper. This paper represents the first comprehensive investigation into the traffic prediction considering both
temporal and spatial shifts. We propose a novel spatio-temporal OOD-network framework with Fréchet Embedding called STONE. STONE effectively models spatial heterogeneity and generates temporal and spatial semantic graphs. Additionally, we introduce a graph perturbation mechanism to enhance the model’s environmental modeling capability for better generalization. We implement extensive experiments on both datasets with spatio-temporal shifts and datasets only with temporal shift, and results demonstrate that STONE achieves competitive performance in terms of both generalization and scalability. 
For rebuttal, we will show the pseudo-code for the Fréchet embedding computation in the STONE model, joint tuning in the training phase and the computation framework of the STONE model, respectively.

<img src='img/Spatial Fréchet Embedding Layer.png' width='240px' alt='The algorithm of Spatial Fréchet Embedding Layer'>

<img src='img/optimization flow.png' width='240px' alt='Optimization flow of STONE during training'>

<img src='img/STONE.png' width='240px' alt='Framework of STONE'>

## 1. Introduction about the datasets
In our experiments, we used SD and GBA datasets which were generated from CA dataset, followed by [LargeST](https://github.com/liuxu77/LargeST/blob/main). For example, you can download CA dataset from the provided [link](https://www.kaggle.com/datasets/liuxu77/largest) and please place the downloaded archive.zip file in the `data/ca` folder and unzip the file. 
### Generating the SD and GBA sub-datasets from CA dataset
First of all, you should go through a jupyter notebook `process_ca_his.ipynb` in the folder `data/ca` to process and generate a cleaned version of the flow data. Then, please go through all the cells in the provided jupyter notebooks `generate_sd_dataset.ipynb` in the folder `data/sd` and `generate_gla_dataset.ipynb` in the folder `data/gla` respectively. Finally use the commands below to generate traffic flow data for our experiments. 
```
python data/generate_data_for_training.py --dataset sd_gba --years 2019_2020_2021
```
## 2. Data Processing

## 3. Environmental Requirments
The experiment requires the same environment as [LargeST](https://github.com/liuxu77/LargeST/blob/main).

## 4. Model Running
To run STONE, for example, you may execute this command in the terminal:
```
bash experiments/stone/run.sh
```
or directly execute the Python file in the terminal:
```
python experiments/stone/main.py --device cuda:0 --dataset SD --years 2019 --model_name stone --seed 0 --bs 64
```
