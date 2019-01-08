#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rodrigo Teles Hermeto

'''Simple script to lunch an experiment'''

import numpy as np
from simulator.engine import Simulation
from simulator.util import print_log,get_all_nodes

training_burst = 6
experiments = np.arange(1,101,1)

# mode 0 = greedy pdr/ mode 1 = greedy j-pdr
for mode in [0,1]:
    for experiment_id in experiments:
        all_nodes = get_all_nodes(experiment_id)
        print_log("Experiment: {}".format(experiment_id))
        for nb_neighbor in [1,2,3,4,5]:
            simulation = Simulation(all_nodes,experiment_id,training_burst)
            s, f, d, t, c, e, i = simulation.start_simulation(nb_neighbor, mode)
            print(s,f,d,t,c,e,i)
