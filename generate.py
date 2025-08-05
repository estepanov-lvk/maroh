from generator.generator import UniformFlowGenerator
from generator.demand_generator import DemandMatrixGenerator
from generator.parsers import sndlib_brain as sndlib_brain_parser
import json
import networkx
import os, sys
from datetime import datetime
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Generate network flows')
    parser.add_argument('topology_path', type=Path,
                       help='path to .gml file with topology, or to directory containing topology.gml')
    parser.add_argument('--mode', choices=['gravity', 'standard'], default='gravity',
                       help='gravity (default) - gravity model, standard - algorithm of synthetic demand generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='random seed for demand generation')
    parser.add_argument('--flows', type=int, default=2,
                       help='Max flow amount parameter. Limits the maximum amount of flows that can exist between any pair of nodes at any time.'
                           'Actual amount of flows is a random number between this and the amount that carried from previous step'
                           'of the generation. So setting this number higher means there will be more flows generated but with'
                           'smaller bandwidths.')
    parser.add_argument('--intensity', type=float, default=0.3,
                        help='Increasing/decreasing this value increases/decreases mean occupied bandwith.'
                        'Choose manually until achieving desired value of mean occupied bandwith (see "average bandwidth taken" in the end of console output).')
                        # TODO: make the selection automatic
    parser.add_argument('--bwvar', type=float, default=3.5,
                        help='Bandwidth variation square root, sets sqrt(max_bw_coef/min_bw_coef) for DemandMatrixGenerator arguments max_bw_coef, min_bw_coef')
    parser.add_argument('--matr', type=int, default=1,
                       help='number of matrices')

    return parser.parse_args()

def generate_from_dataset():
    # create generator and set its parameters
    generator = UniformFlowGenerator(5, 400000, 2000000)

    # parse folder that contains dataset files
    matrices = sndlib_brain_parser.parse_all('generator/dataset/')

    # run the generator
    result = generator.generate(matrices, 60000)

    # convert results to a string with optional pretty print - can be removed, only result.json() is needed
    # But results will be unreadable for a human is pretty print is removed
    pretty_res = json.dumps(json.loads(result.json()), indent=4)

    # open result file and write the flow data
    with open('flows.log', 'w') as f:
        f.write(pretty_res)


def generate_synthetic(topology_file_path: str,
                       out_file_path: str,
                       mode: str,
                       n_matrices: int,
                       max_flow_amount: int,
                       intensity: float,
                       bw_variation_sqrt: float,
                       seed: int = None):
    # get a topology
    with open(topology_file_path, mode='rb') as f:
        topology = networkx.readwrite.read_gml(f)

    coef = 1.3 # may need to change manually for bw_variation_sqrt values other than 3.5
    mean_bw_expected = intensity / (bw_variation_sqrt * coef)
    min_bw_coef = mean_bw_expected / bw_variation_sqrt
    max_bw_coef = mean_bw_expected * bw_variation_sqrt
    generator = DemandMatrixGenerator(min_bw_coef, max_bw_coef, topology, mode=mode, seed=seed)

    # generate some matrices
    # matrices = generator.generate(50, ['1', '2', '3', '4'], ['13', '14', '15', '16'])
    matrices = generator.generate(n_matrices)

    # generate flows using uniform flow generator
    flow_generator = UniformFlowGenerator(max_flow_amount, 25000, 700000)

    # run the generator
    result = flow_generator.generate(matrices, 30000)

    # convert results to a string with optional pretty print - can be removed, only result.json() is needed
    # But results will be unreadable for a human is pretty print is removed
    pretty_res = json.dumps(json.loads(result.json()), indent=4)

    # open result file and write the flow data
    with open(out_file_path, 'w') as f:
        f.write(pretty_res)

if __name__ == '__main__':
    args = parse_args()

    # set topology_path as path to topology.gml file
    topology_path = args.topology_path
    if topology_path.is_dir():
        topology_path = topology_path / "topology.gml"

    # name of directory containing topology.gml
    topology_name = topology_path.resolve().parent.name
    dat_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
    out_file_path = f"flows_{topology_name}_{dat_str}.json"

    mode = args.mode
    seed = args.seed
    n_matrices = args.matr
    max_flow_amount = args.flows
    intensity = args.intensity
    bw_variation_sqrt = args.bwvar

    print(f"Topology path: {topology_path}")
    print(f"Output file: {out_file_path}")
    generate_synthetic(topology_path, out_file_path, mode, n_matrices,
        max_flow_amount, intensity, bw_variation_sqrt, seed=seed)
