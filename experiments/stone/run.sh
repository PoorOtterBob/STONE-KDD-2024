## Spatial Temporal OOD ##
# SD 2019
# python experiments/stone/main.py --device cuda:0 --dataset SD --years 2019 --model_name stone --seed 0 --bs 64 --c 10 --ood 1 --tood 1
# SD 2020
# python experiments/stone/main.py --device cuda:0 --dataset SD --years 2020 --model_name stone --seed 0 --bs 64 --c 10 --ood 1 --tood 1
# GBA 2019
# python experiments/stone/main.py --device cuda:0 --dataset GBA --years 2019 --model_name stone --seed 0 --bs 64 --c 30 --ood 1 --tood 1
# GBA 2020
# python experiments/stone/main.py --device cuda:0 --dataset GBA --years 2020 --model_name stone --seed 0 --bs 64 --c 30 --ood 1 --tood 1


## Only Temporal OOD ##
# SD 2019
# python experiments/stone/main.py --device cuda:0 --dataset SD --years 2019 --model_name stone --seed 0 --bs 64 --c 10 --ood 0 --tood 1
# SD 2020
# python experiments/stone/main.py --device cuda:0 --dataset SD --years 2020 --model_name stone --seed 0 --bs 64 --c 10 --ood 0 --tood 1
# GBA 2019
# python experiments/stone/main.py --device cuda:0 --dataset GBA --years 2019 --model_name stone --seed 0 --bs 64 --c 30 --ood 0 --tood 1
# GBA 2020
# python experiments/stone/main.py --device cuda:0 --dataset GBA --years 2020 --model_name stone --seed 0 --bs 64 --c 30 --ood 0 --tood 1


## Non-OOD ##
# SD 2019
# python experiments/stone/main.py --device cuda:0 --dataset SD --years 2019 --model_name stone --seed 0 --bs 64 --c 10 --ood 0 --tood 0
# SD 2020
# python experiments/stone/main.py --device cuda:0 --dataset SD --years 2020 --model_name stone --seed 0 --bs 64 --c 10 --ood 0 --tood 0
# GBA 2019
# python experiments/stone/main.py --device cuda:0 --dataset GBA --years 2019 --model_name stone --seed 0 --bs 64 --c 30 --ood 0 --tood 0
# GBA 2020
# python experiments/stone/main.py --device cuda:0 --dataset GBA --years 2020 --model_name stone --seed 0 --bs 64 --c 30 --ood 0 --tood 0

