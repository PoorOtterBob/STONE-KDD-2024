import argparse

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    parser.add_argument('--years', type=str, default='2019')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=2018)

    parser.add_argument('--bs', type=int, default=64)
    # seq_len denotes input history length, horizon denotes output future length
    # Model for ST-Block
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)

    # For data segment
    parser.add_argument('--order', type=int, default=0) # 0-displacement; 1-courier; 2-acceleration and consider the total number 1+2*order (useless)
    parser.add_argument('--new_node_ratio', type=float, default=0.1)
    parser.add_argument('--num_val', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=0.01) # Measure similarity between points for Hamming Metric (useless)
    parser.add_argument('--c', type=int, default=1) # The number of rounds of Anchorset repetitive sampling in Bourgain's theorem


    # For solving equation
    parser.add_argument('--basis_num', type=int, default=1) # like horizon
    parser.add_argument('--KK', type=int, default=2) # num of mask
    parser.add_argument('--num_sample', type=float, default=0.01) # ratio of mask

    # for traning
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--test_ratio', type=float, default=0.1) # [0.05, 0.1, 0.15, 0.2]
    parser.add_argument('--ood', type=int, default=1)
    parser.add_argument('--tood', type=int, default=1) # Activated when 
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--alpha', type=int, default=0.8) # (useless)
    parser.add_argument('--beta', type=int, default=0.1) # (useless)
    parser.add_argument('--beta0', type=int, default=2) # L = L_ob + gammaL_un' (useless)
    return parser


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
