from networkx import product
import sys
from sys import stdout as out
import gurobipy as gp
import networkx as nx
import json
import time
import numpy as np
import pandas as pd
from collections import Counter

def get_arg(i):
    return sys.argv[i] if len(sys.argv) > i else None

def zero_nconnect(model, q, N, V, Bandwidth):
    """
    Constaint for the model , the condition of equality to zero of the sum of he values  of the binary function for vertices,
    between which there are no links
    """
    model.addConstr(gp.quicksum((q[i, j, n]) for (i, j, n) in product.product(V, V, N) if Bandwidth.at[i,j] == 0) == 0)


def eval_cnstr1(model, q, I, F, N, M, V):
    """
    Condition for the flow's starting point
    """
    for k in N:
        a_n = F[k]['start']
        for m in M:
            if I[m]['flows_on_interval']:
                if k in I[m]['flows_on_interval']:
                    model.addConstr(gp.quicksum((q[a_n, j, k]) for j in V) == 1)


def eval_cnstr2(model, q, I, F, N, M, V):
    """
    Condition for the flow's final point
    """
    for k in N:
        b_n = F[k]['end']
        for m in M:
            if I[m]['flows_on_interval']:
                if k in I[m]['flows_on_interval']:
                    model.addConstr(gp.quicksum((q[j, b_n, k]) for j in V) == 1)


def eval_cnstr3(model, q, I, F, M, V):
    """
    Flow conservation
    """
    for m in M:
        if I[m]['flows_on_interval']:
            for k in I[m]['flows_on_interval']:
                a_i = F[k]['start']
                b_j = F[k]['end']
                for r in V - {a_i, b_j}:
                    sum_out = gp.quicksum((q[r, j, k]) for j in V)
                    sum_in = gp.quicksum((q[i, r, k]) for i in V)
                    model.addConstr(sum_in == sum_out)

def eval_cnstr4(model, q, I, F, M, V):
    """
    Constraint on the flow route
    """
    for m in M:
        if I[m]['flows_on_interval']:
            for k in I[m]['flows_on_interval']:
                a_i = F[k]['start']
                b_j = F[k]['end']
                for r in V - {a_i, b_j}:
                    model.addConstr((gp.quicksum((q[r, j, k]) for j in V) + gp.quicksum((q[i, r, k]) for i in V)) <= 2)


def eval_cnstr5(model, q, M, I, F, V, Bandwidth):
    """
    Bandwidth limitation
    """
    for m in M:
        for i in V:
            for j in V:
                sumx = 0
                if Bandwidth.at[i,j]:
                  if I[m]['flows_on_interval']:
                       for k in I[m]['flows_on_interval']:
                           for r in F[k]['all_bandwidth']:
                                if ((int(r) >= I[m]['0'] and int(r) <= I[m]['1']) or int(r) >= I[m]['1']):
                                   sumx += q[i,j,k] * F[k]['all_bandwidth'][r]
                model.addConstr(sumx <= int(Bandwidth.at[i,j]))


def eval_cnstr6(model, q, V, N):
    """
    Binary function non-negativity condition
    """
    for (i, j, n) in product.product(V, V, N):
        model.addConstr(q[i,j,n] >= 0)

def eval_cnstr7(model, q, I, F, N, M, V):
    """
    Condition to prevent flow return into its starting point
    """
    for k in N:
        b_n = F[k]['start']
        for m in M:
            if I[m]['flows_on_interval']:
                if k in I[m]['flows_on_interval']:
                    model.addConstr(gp.quicksum((q[j, b_n, k]) for j in V) == 0)


def eval_cnstr8(model, q, I, F, N, M, V):
    """
    Condition to prevent flow going out of its final point
    """
    for k in N:
        a_n = F[k]['end']
        for m in M:
            if I[m]['flows_on_interval']:
                if k in I[m]['flows_on_interval']:
                    model.addConstr(gp.quicksum((q[a_n, j, k]) for j in V) == 0)


def set_time(t0, T, flows):
    """
    Function for time intervals counting (since the flow rate value changes)
    Used for the calculation of the current load of each link  
    """
    t = {}
    points = set()
    points.add(t0)
    points.add(T)
    for f in flows:
        for key in f['all_bandwidth']:
            points.add(int(key))
    points = sorted(list(points))
    for i in range(len(points) - 1):
        e = points[i+1] - points[i]
        C = set()
        for c in range(len(flows)):
            if points[i] >= flows[c]['start_time'] and points[i] <= flows[c]['end_time']:
                C.add(c)
        t[i] = {'0': points[i],'1': points[i + 1],'interval': e, 'flows_on_interval': C}
    return points, t


if __name__ == "__main__":
    """
    Reading the topology
    """
    topo_read = nx.read_gml(get_arg(1) or "topology.gml",destringizer=int)
    topo_convert = nx.convert_node_labels_to_integers(topo_read)
    topo_dict = nx.to_dict_of_dicts(topo_convert)
    number_of_edges = topo_convert.number_of_edges()
    nodes = list(topo_convert.nodes)
    nodenum = len(nodes)
    edgesdict = []
    for i in range(nodenum):
        for k in topo_dict[nodes[i]].keys():
            for r in range(len(topo_dict[nodes[i]][k])):
                edgesdict.append({"A": nodes[i], "B": k, "R": topo_dict[nodes[i]][k][r]['bandwidth'], "RK": topo_dict[nodes[i]][k][r]['current_bandwidth']})


    """
    Creating a dataframe with link bandwidth information
    """
    Bandwidth = pd.DataFrame(0, index = np.arange(len(topo_convert.nodes)),columns = np.arange(len(topo_convert.nodes)))
    for edge in edgesdict:
        Bandwidth[edge['B']][edge['A']] = edge['R']
        Bandwidth[edge['A']][edge['B']] = edge['R']


    """
    Reading a flow information file
    """
    with open(get_arg(2) or 'flows.json', 'r') as f:
        flows = json.load(f)
    dictnodes = dict(zip(list(topo_read.nodes),nodes))

    if int(get_arg(3) or '1') != 0:
        for i in flows:
            if type(i['start']) == str:
                i['start'] = dictnodes[int(i['start'])]
                i['end'] = dictnodes[int(i['end'])]
            else:
                i['start'] = dictnodes[i['start']]
                i['end'] = dictnodes[i['end']]
    else:
        i['start'] = dictnodes[i['start']]
        i['end'] = dictnodes[i['end']]
    V = set(range(nodenum))


    """
    Setting the time interval for receiving data
    """
    t0 = 0
    T = 125000
    timeline,Intervals = set_time(t0, T, flows)
    M = set(range(len(Intervals)))


    """
    Creating a model, a binary function (that is a variable in the context of the model)
    applying all constraints to the received data
    """
    model = gp.Model()
    N = set(range(len(flows)))
    q = { (i,j,n) : model.addVar(name = "q[%s,%s,%s]" % (i,j,n), vtype = gp.GRB.BINARY) for i in V for j in V for n in N }
    model.update()
    zero_nconnect(model,q,N,V,Bandwidth)
    eval_cnstr1(model, q, Intervals, flows, N, M, V)
    eval_cnstr2(model, q, Intervals, flows, N, M, V)
    eval_cnstr3(model, q, Intervals, flows, M, V)
    eval_cnstr4(model, q, Intervals, flows, M, V)
    eval_cnstr5(model, q, M, Intervals, flows, V, Bandwidth)
    eval_cnstr6(model, q, V, N)
    if not (int(get_arg(4) or '0') != 0):
        eval_cnstr7(model, q, Intervals, flows, N, M, V)
        eval_cnstr8(model, q, Intervals, flows, N, M, V)


    """
    Calculation of the current load for each link based on the model
    """
    count = 0
    CBandwidth = pd.DataFrame(0, index = np.arange(len(topo_convert.nodes)),columns = np.arange(len(topo_convert.nodes)))
    pd.options.mode.chained_assignment = None
    blist = list()
    for m in M:
        for i in V:
            for j in V:
              if Bandwidth.at[i,j] != 0:
                  if Intervals[m]['flows_on_interval']:
                      cur_b1 = []
                      for k in Intervals[m]['flows_on_interval']:
                          cur_b = 0
                          for r in flows[k]['all_bandwidth']:
                              if (int(r) >= Intervals[m]['0'] and int(r) <= Intervals[m]['1']) or (int(r) == max(flows[k]['all_bandwidth']) and int(r) <= Intervals[m]['1']):
                                  cur_b += q[i, j, k]*flows[k]['all_bandwidth'][r]    
                          cur_b1.append(cur_b)
                      CBandwidth[i][j] = sum(cur_b1)
                      count += 1
        blist.append(CBandwidth)


    """
    Calculation of the phi-function value and search for the optimum
    """
    phi_list = list()
    phivar = model.addVar(vtype = gp.GRB.CONTINUOUS, name = 'phivar')
    links_list = [(i,j) for (i,j) in product.product(V, V) if Bandwidth.at[i,j] != 0]
    NN = len(links_list)
    for m in M:
        average = gp.quicksum((blist[m].at[i,j] / Bandwidth.at[i,j]) for i in V for j in V if Bandwidth.at[i,j]!=0) / (NN)
        model.addConstr((gp.quicksum(((blist[m].at[i,j] / Bandwidth.at[i,j] - average) * (blist[m].at[i,j] / Bandwidth.at[i,j] - average)) for i in V for j in V if Bandwidth.at[i,j]!=0)/(NN))<=phivar)
    model.setObjective(phivar, gp.GRB.MINIMIZE)
    model.optimize()
    model.write("model.lp")


    """
    debugging, to output the values of binary functions
    debugging, to display, how the program has distributed flows in topology
    """
    all_vars = model.getVars()
    values = model.getAttr("X", all_vars)
    names = model.getAttr("VarName", all_vars)

    #for name, val in zip(names, values):
    #    print(f"{name} = {val}")
    

    """
    Weights, based on number of flows on current link
    """
    list_vars = list(zip(names, values))
    list_w = list()
    for i in range(len(list_vars) - 2):
        if list_vars[i][1] == 1:
            list_w.append(list_vars[i])
    coord = list()
    for i in list_w:
        coord.append(tuple(map(int, i[0][2:(len(i[0])-1)].split(',')))[0:2])
    weights = Counter(coord)
    for c, w in weights.items():
        print(f"Weight {c} = {w}")

