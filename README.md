# Input data format explanation

### command line arguments

To run SAMAROH or SAMAROH-2L, run `main.py dir --multi`, where dir is path to directory containing experiment input data (config.yaml, flows.json, topology.gml, topology_changes.json, logging.yaml files). config.yaml specifies whether to use SAMAROH or SAMAROH-2L in this case (use_memory: False or True).
To run MAROH or MAROH-2L, run `main.py dir`. config.yaml specifies whether to use MAROH or MAROH-2L in this case (use_memory: False or True).
Directories with experiment input data can be found in data_examples directory. Specific topologies, flows and config.yaml parameters used for experiments for the article can be found there too.

### topology

Topology file is in gml format. Current examples are synthetic or taken from topology zoo and modified. Link parameters "bandwidth" and "current_bandwidth" were added manually.

### topology_changes

This file holds how topology changes over time in json format.  
Dict keys are moments of time in milliseconds when the change is detected. This means that if, for example, there are dict keys "10000" and "50000", then at any moment of time between 10000ms and 50000ms the topology has changes described by "10000" dict.  
Inside the change dict are two lists. They describe which nodes and links are missing. Missing node automatically means that all its links are also missing. If both lists are empty, it means there are no changes to topology.

### flows

This json is a list of all flows present throughout the experiment. Each flow has start and end time in milliseconds. Flow bandwidth is represented as a dict similar to topology changes: dict keys are moments of time when flow bandwidth changes

### config.yaml

This file holds config for the experiment input data located in the folder. Values explanation:  
**lsdb_period** - model time interval between algorithm iterations. Note: if topology change was detected, time between iterations may be shorter than period, but never longer  
**plot_period** - plot phi graph each time this amount of episodes passes  
**iterations** - amount of algorithm iterations  
**log_path** - path to output file with log  
**log_level** - logging level  
**hash_function, algorithm, path_calculator** - python import paths to implementation of the related component. These paths must be importable from main.py.  
**phi** - python import path to phi function that should be used. These paths must be importable from main.py. There are two available: dte_stand.phi_calculator.PhiCalculator.calculate_phi and dte_stand.phi_calculator.PhiCalculator.calculate_max_value  

**mate** - section related to MateAlgorithm  
 - **episodes** - number of episodes
 - **horizons** - number of horizons per episode (amount of weight change events, i.e. how long the trajectory is)
 - **gamma**, **gae_lambda** - reward estimation parameters
 - **min_weight**, **max_weight** - minimum and maximum random weights the algorithm starts with
 - **reward** - how to calculate reward. This parameter is passed to Environment's **base_reward** parameter. It can be 'phi', 'min_max' or anything else the algorithm supports (refer for **compute_reward_measure** function)
 - **reward_computation** - how reward is assigned to the algorithm. It can be either 'change' (difference between previous reward and current reward) or 'value' (negative raw reward value)
 - **greedy_epsilon** - value of epsilon for epsilon-greedy strategy. Usually set as something around 0.8-0.95
 - **message_iterations** - how many iterations of message passing to do. If not given, defaults to graph's diameter
 - **actions** - config for actions available to mate algorithm. Actions affect the weight of a link in the graph. Possible actions: addition, subtraction, multiplication, division, zero. Action is true of false and defines whether this action eis available to algorithm. Value defines amount of addition/subtraction/multiplication/division.
 - **n_without_update** - period (in episodes) of neural networks updating (using history of states, actions, rewards, etc accumulated since the previous update).
 - **lr_actor** - actor's learning rate (parameter for Adam optimizer).
 - **lr_critic** - critic's learning rate (parameter for Adam optimizer).
 - **actor_cfg** - optional actor configuration. If not defined, default values will be used, which can be checked in `[ActorName]Cfg` classes and may be different for different actor implementations. Currently has following fields:
    - **use_gat** - whether to use GAT (graph attention network) in actor instead of MPNN, default: false;
    - **gat_num_heads** - number of attention heads in GAT;
    - **use_memory** - whether to use memory, default: false;
    - **memory_size** - number of states saved in memory;
    - **threshold** - threshold, which agent's memory uses to determine whether received state is close enough to any state in memory to be used instead of performing message passing iteration;
    - **clustering** - clustering method used for defragmentating memory when it's full (MiniBatchKMeans, Agglomerative);
    - **metric** - distance metric, which agent's memory uses to measure distance between states (l2, l1, cosine).

### logging.yaml

This file contains logging config. Refer to standard python module "logging", function "dictConfig" for format reference

# Extension guide

There are 3 replaceable components in the stand: hash function, hash-weights calculation algorithm and path calculator. They are located in corresponding folders inside the dte_stand folder.    
Every folder has a base.py file which contains a base class for a component, and a dummy.py file which contains a simple example of component class's interfaces.  
In order to create a new component implementation, you need to inherit from the base class and implement the one abstract function that exists in the base class.

Note: interfaces are not final! any suggestions and fixes are welcome

Current interfaces are described below.

### Algorithm

Algorithm has a single abstract function "step" which is expected to perform one iteration of the algorithm. This function gets topology as an input parameter and must return a HashWeights object. Its structure and methods can be found in data_structures/hash_weights.py

### Hash function

Hash function has one function to implement: _choose_nexthop which must choose one nexthop. As parameters it receives a list of bucket objects (located in data_structures/hash_weights.py). Bucket represents an edge in the graph and its hash-weight. Only buckets that are allowed to be chosen (according to the paths that were build by path calculator) are present in the list. Edge in the graph is represented as GraphPathElement object. Hash function must return a GraphPathElement object from the chosen bucket.

### Path calculator

This component calculates the list of available paths in the graph. Parameters: topology, source node, destination node. Return value: list of paths, where one path is a list of GraphPathElement objects (GraphPathElement object corresponds to an edge in the graph)

# How to use generators

Folder 'generator' contains classes that generate the list of flows from traffic matrices. Currently only one generator is implemented: uniform.
Folder 'parsers' contains traffic matrix parsers for different datasets. Currently only one dataset is implemeted: sndlib's brain dataset

file generate.py can be used to run the generator. To do so, dataset files should be unpacked into a separate folder. Any number of files from the dataset can be taken. Each file contains a single traffic matrix, so the more files are present, the more data points is available to generator, so generated input data will cover a longer experiment. Files taken from the dataset must be sequential.
The following parameters can be set in generate.py:
 - generator parameters - refer to chosen generator's documentation in the code
 - input folder where dataset files are located
 - period between data points in the dataset. Refer to dataset description to get this value. Although from the generator's standpoint it is not required to set period according to dataset. If the dataset specifies 1 minute interval between data points, in code it is allowed to set it to any other value you want. The data will be interpreted according to the period you specified.
 - output file path

Resulting flow bandwidth is just a number generated according to what is given in the dataset. For example for sndlib brain dataset it is bits per second. Other dataset may use different scale. Link bandwidth in topology should be given according to the scale used in the dataset.

demand_generator.py is a generator of random demand matrices for given topology. These demand matrices can then be fed into any of the flow generators to generate a list of flows. An example of this is given in function generate_synthetic() in generate.py

Flows used for the article were generated using `generate.py` with parameters `--matr 1`, different values of `--flows`, `--intensity`, `--seed`, and the rest of parameters having default values.

# How to use converter

Converter does two things:
 - duplicates every link and orients them in different directions (A-B becomes A->B, B->A)
 - adds parameters "bandwidth" and "current_bandwidth" to links. current_bandwidth should always be 0 in topology file, bandwidth can be set later as you need. If link has "id" parameter (topology zoo topologies do), then one of the links will have "_r" suffix added.

converter accept 2 parameters - path to original file and path where to save the result

# How to use genetic algorithm

Run `optimalweights.py dir`, where dir is path to directory containing topology.gml and flows.json. It will find set of agent weights with sub-optimal Ф value.
 - --iter - number of iterations of genetic algorithm. Increasing this may result to a solution closer to optimal but also a proportional increase in execution time.
 - --maxweight specifies maximum value of agent weight, while algorithm always assigns agent weights integer numbers between 1 and maxweight.
 - --ntrain: if value is not specified (so it is 1 by default), the resulting solution will be sub-optimal for the same random seed of hash function as used in MAROH. If value more than 1 specified, the algorithm will measure objective function of each solution as Ф averaged by multiple alternative random seeds of hash function (multiple tries of distributing flows using the same agent weights).
 - --ntest value specifies number of extra alternative random hash seeds on which Ф will be measured for best solution on each iteration, but not used for algorithm's decisions.
 - --nvalid value specifies number of extra alternative random hash seeds on which Ф will be measured on best solution after algorithm finishes.
Note: Genetic algorithm can only provide solution for a static set of flows. If flows.json contain different bandwidths at different timestamps, specify --flowsperiod, --start, --end to make sure that genetic algorithm use only one specific set of flows from the file.


# How to draw plots like in the article

Run `plot_advanced.py path_to_json`, where path_to_json is path to json file containing list of experiment results directories and their names, plot title, (optionally) genetic algorithm result csv file. See plot_advanced_input_example.json for format example. For example, if you ran `main.py data_examples/topology ...`, experiment result directory path will be `data_examples/topology/exp_...`, which can be specified as "path" value in json file for plotting.
