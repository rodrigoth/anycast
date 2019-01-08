#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rodrigo Teles Hermeto

import numpy as np
import pandas as pd
import sys
from simulator.node import Node
from simulator.schedule import Slotframe
from simulator.packet import Packet
from simulator.neighbor import Neighbor
from simulator.util import print_log,get_database_connection
from collections import defaultdict
import random

class Simulation:
    def __init__(self,sink,training_burst_size,nb_parents,number_of_tries=3):
        sink = Node(sink)
        sink.rank = 256

        self.nodes = [sink]
        self.training_burst_size = training_burst_size

        self.slotframe = Slotframe()
        self.number_of_tries = number_of_tries
        self.parent_stats = []
        self.nb_parents = nb_parents
        self.stability_dic = defaultdict(lambda: defaultdict(list))


    def build_topology(self):
        sink = self.nodes[0]
        sink.rank = 256

        self.add_child(sink)

        [self.add_child(node) for node in self.nodes if node.rank == 512]
        [self.add_child(node) for node in self.nodes if node.rank == 768]


    def update_ranks(self):
        for node in self.nodes[1:]:
            min_rank = 65000
            for receivers in self.get_all_receivers(node):
                rec = next(x for x in self.nodes if x.id == receivers)
                attempt_rank = self.compute_rank(node,rec)
                if attempt_rank < min_rank:
                    node.rank = attempt_rank
                    node.parents = [rec]
                    min_rank = attempt_rank



    def compute_rank(self,node,neighbor):
        query = "select count(1) from anycast_transmissions where sender = '{}' and node = '{}' and burst <= {}".format(node.id,neighbor.id,self.training_burst_size)
        conn = get_database_connection()
        cur = conn.cursor()
        cur.execute(query)
        row = cur.fetchone()

        acked = row[0]

        if acked == 0:
            return neighbor.rank + (3*4 - 2)*256
        return neighbor.rank + (3*(float(120)/acked) - 2) * 256


    def add_child(self, node):
        neighbors = self.get_all_receivers(node)
        for nei in neighbors:
            if nei not in [n.id for n in self.nodes]:
                new_node = Node(nei)
                new_node.rank = node.rank + 256
                self.nodes.append(new_node)

    def print_topology(self):
        print(self.nodes[0])
        print()

        for node in self.nodes:
            if node.rank == 512:
                print(node)

        print()

        for node in self.nodes:
            if node.rank == 768:
                print(node)

        print()

        for node in self.nodes:
            if node.rank == 1024:
                print(node)

    def get_all_receivers(self,sender):
        conn = get_database_connection()
        query = "select distinct node from anycast_transmissions where sender = '{}' ".format(sender.id)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        return [row[0] for row in rows]


    def fill_nodes_transmission_queue(self, nb_packets):
        senders = self.nodes[1:] # exclude sink
        for sender in senders:
            if sender.rank == 1024:
                packets = []
                for pkt in np.arange(1, nb_packets + 1, 1):
                    packets.append(Packet(sender, pkt))
                sender.queue = packets

    def print_transmission_queue(self, node_id):
        node = next((x for x in self.nodes if x.id == node_id), None)
        for packet in node.queue:
            print("Node: {}, payload {}".format(packet.node.id, packet.payload))

    def print_slotframe(self):
        for cell in self.slotframe.cells:
            print(cell)

    def allocate_cell(self, timeslot, channel, node, receivers):
        cell = next((x for x in self.slotframe.cells if x.timeslot == timeslot and x.channel == channel), None)
        if not cell.used:
            cell.sender = node
            cell.receivers = receivers
            cell.used = True
            return True
        return False

    def build_slotframe(self, nb_parents, mode,training_set_id):
        channel = 0
        seq = np.arange(0, 101, 1)

        timeslot_index = 0

        for node in self.nodes[1:]:
            if node.rank == 512:
                receivers = [self.nodes[0]]
            else:
                receivers, avg_correlation = self.get_best_parents(nb_parents, mode, node,training_set_id)

            self.allocate_cell(seq[timeslot_index], channel, node, receivers)
            timeslot_index += 1

        #return avg_correlation
        return 0


    def get_best_parents(self, nb_parents, mode, node,training_set_id):
        def compute_j_pdr(selected_parents):
            if len(selected_parents) == 1:
                return 1 - (len(selected_parents[0].sequence) - np.count_nonzero(selected_parents[0].sequence)) / len(
                    selected_parents[0].sequence)
            else:
                result = np.zeros(len(selected_parents[0].sequence))
                for parent in selected_parents:
                    result = np.add(result, parent.sequence)

            return 1 - ((len(result) - np.count_nonzero(result)) / len(result))

        def get_next_parent(selected_parents, neighbors):
            max_j_pdr = 0
            best_neighbor = None
            for nei in neighbors:
                if not nei in selected_neighbors:
                    selected_parents.append(nei)
                    jpdr = compute_j_pdr(selected_parents)
                    if jpdr > max_j_pdr:
                        max_j_pdr = jpdr
                        best_neighbor = nei
                    selected_parents.pop()
            return best_neighbor

        def compute_phi(array_a, array_b):
            sum_array = np.add(array_a, array_b)
            n11 = sum_array.tolist().count(2)
            n00 = sum_array.tolist().count(0)

            n10 = 0
            n01 = 0
            for a, b in zip(array_a, array_b):
                if a == 1 and b == 0:
                    n10 += 1
                else:
                    if a == 0 and b == 1:
                        n01 += 1

            n01_n00 = n01 + n00
            if n01_n00 == 0:
                n01_n00 = 1

            n11_n10 = n11 + n10
            if n11_n10 == 0:
                n11_n10 = 1

            n10_n00 = n10 + n00
            if not n10_n00:
                n10_n00 = 1

            n11_n01 = n11 + n01
            if not n11_n01:
                n11_n01 = 1

            return float(n11 * n00 - n10 * n01) / np.sqrt(n11_n10 * n01_n00 * n10_n00 * n11_n01)

        def compute_k_factor(array1, array2):
            phi = compute_phi(array1, array2)

            px = np.sum(array1) / len(array1)
            py = np.sum(array2) / len(array2)

            sigmax = np.sqrt(px * (1 - px))
            sigmay = np.sqrt(py * (1 - py))

            if phi > 0:
                max_p = (np.minimum(px, py) - px * py) / (sigmax * sigmay)
                result = phi / max_p
            else:
                if not sigmax:
                    sigmax = 1
                    px = 1
                if not sigmay:
                    sigmay = 1
                    py = 1

                min_p = (-px * py) / (sigmax * sigmay)
                result = phi / min_p

            if result >= 1.0:
                result = 0.999329

            return result


        selected_neighbors = []

        neighbors_lower_rank = [n for n in node.neighbors if n.node.rank < node.rank]

        # 1. Gready pdr

        neighbors_lower_rank.sort(key=lambda x: x.pdr, reverse=True)

        current_index = 0
        selected_neighbors.append(neighbors_lower_rank[current_index])

        if mode == 0:
            while len(selected_neighbors) < nb_parents and current_index + 1 < len(neighbors_lower_rank):
                current_index += 1
                selected_neighbors.append(neighbors_lower_rank[current_index])

        if mode == 1:
            while len(selected_neighbors) < nb_parents and current_index + 1 < len(neighbors_lower_rank):
                current_index += 1
                p = get_next_parent(selected_neighbors, neighbors_lower_rank)
                if p is not None:
                    selected_neighbors.append(get_next_parent(selected_neighbors, neighbors_lower_rank))

        selected_neighbors.sort(key=lambda x: x.pdr, reverse=True)

        mean = 0
        fisher_correlations = []

        if len(selected_neighbors) > 1:
            fisher_correlations = []
            for i in range(0, len(selected_neighbors)):
                for j in range(i + 1, len(selected_neighbors)):
                    if not np.isnan(compute_k_factor(selected_neighbors[i].sequence, selected_neighbors[j].sequence)):
                        fisher_correlations.append(np.arctanh(compute_k_factor(selected_neighbors[i].sequence, selected_neighbors[j].sequence)))
                    else:
                        fisher_correlations.append(0)

        if fisher_correlations:
            mean = np.tanh(np.mean(fisher_correlations))

        selected_neighbors = [nei.node for nei in selected_neighbors]

        return selected_neighbors, mean

    def get_training_burst(self,sender,receivers,begin,end):
        query = "select min(seqnum), max(seqnum) from anycast_transmissions where burst between {} and {} and sender = '{}'".format(begin,end,sender.id)

        conn = get_database_connection()
        cur = conn.cursor()
        cur.execute(query)
        row = cur.fetchone()
        seqnum_lst = np.arange(row[0], row[1] + 1, 1)
        conn.close()

        anycast_tx = pd.DataFrame(0, index=[node for node in receivers], columns=seqnum_lst.tolist())

        query = "select seqnum,node from anycast_transmissions where burst between {} and {} and sender = '{}' " \
                " order by seqnum".format(begin,end, sender.id)
        conn = get_database_connection()
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        for row in rows:
            anycast_tx.set_value(row[1], row[0], int(1))

        anycast_tx = anycast_tx[anycast_tx.sum(axis=1) / len(anycast_tx.columns) < 0.75]

        return anycast_tx

    def update_neighbors(self,begin,end):
        for node in self.nodes[1:]:
            if node.rank == 512:
                new_neighbor = Neighbor(self.nodes[0])
                node.neighbors.append(new_neighbor)
            else:
                all_receivers = self.get_all_receivers(node)
                training_burst = self.get_training_burst(node,all_receivers,begin,end)
                for receivers in all_receivers:
                    if receivers in training_burst.index:
                        rec = next(x for x in self.nodes if x.id == receivers)
                        new_neighbor = Neighbor(rec)
                        new_neighbor.sequence = list(training_burst.loc[rec.id, :].values)
                        new_neighbor.pdr = np.sum(new_neighbor.sequence) / len(new_neighbor.sequence)
                        node.neighbors.append(new_neighbor)

    def was_received_by_one_parent(self, sender,receivers,all_rx):
        result = []
        aux = []
        rx = all_rx[sender.id]

        key = random.choice(list(rx.keys()))

        value = rx[key]

        for res in list(set([r.id for r in receivers]).intersection(value)):
            rec = next(x for x in self.nodes if x.id == res)
            aux.append(rec)

        for ordered_receivers in receivers:
            for node in aux:
                if ordered_receivers.id == node.id:
                    result.append(node)

        return result

    def get_all_rx_by_nodes(self):
        query = "select sender,seqnum,node from anycast_transmissions where burst > {} order by node,burst,seqnum".format(self.training_burst_size)
        conn = get_database_connection()
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()

        dic_rx = defaultdict(lambda: defaultdict(list))
        for row in rows:
            dic_rx[row[0]][row[1]].append(row[2])

        return dic_rx

    def get_idle_freq(self,receivers,all_receivers, nb_neighbors):
        nei = list(np.zeros(nb_neighbors))
        index = 0

        for rec in all_receivers:
            if rec.id in receivers:
                nei[index] = 1
            index += 1

        self.parent_stats.append(nei)

        return 0

    def has_packet_to_transmit(self):
        for node in self.nodes[1:]:
            if len(node.queue) > 0:
                return True
        return False

    def prepare_simulation(self, mode,begin,end,training_set_id):
        self.build_topology()
        self.print_topology()
        self.update_neighbors(begin,end)
        self.build_slotframe(self.nb_parents, mode,training_set_id)


    def start_simulation(self):
        current_cell_index = 0
        asn = 1
        success = 0
        fail = 0
        delays = []
        total_transmissions = 0
        total_rx = 0
        energy = 0
        idle = 0

        duplicated_packets = 0
        sink_received = 0

        all_receptions = []

        self.fill_nodes_transmission_queue(1000)

        while self.has_packet_to_transmit():
            current_cell = self.slotframe.cells[current_cell_index]

            if current_cell.used:
                packet = current_cell.sender.get_next_packet()
                if packet is not None:
                    if not all_receptions:
                        all_receptions = self.get_all_rx_by_nodes()

                    # first try
                    if packet.attempts_left == self.number_of_tries and not packet.is_forwarding:
                        packet.sent_asn = asn

                    total_transmissions += 1

                    parents_that_received = self.was_received_by_one_parent(current_cell.sender, current_cell.receivers,all_receptions)
                    result = len(parents_that_received) > 0

                    if result:
                        # energy sender
                        energy += 74.99
                        already_forwarded = False

                        for receiver in current_cell.receivers:
                            if receiver.id in [par.id for par in parents_that_received]:
                                # sink received
                                if receiver.id == self.nodes[0].id:
                                    packet.received_asn = asn
                                    delays.append(asn - packet.sent_asn)
                                    sink_received += 1
                                else:
                                    packet.is_forwarding = True
                                energy += 78.16  # rx + ack

                                if not already_forwarded:
                                    packet.number_of_tries = 3
                                    receiver.queue.append(packet)
                                    already_forwarded = True
                            else:
                                # idle listening
                                energy += 31.98

                        current_cell.sender.remove_transmitted_packet()
                    else:

                        fail += 1
                        packet.attempts_left -= 1

                        # energy sender
                        energy += 70.316

                        for receiver in current_cell.receivers:
                            energy += 31.98

                        if packet.attempts_left == 0:
                            current_cell.sender.remove_dropped_packet()

                    idle_nodes = 100 - 1 + len(current_cell.receivers)
                    energy += idle_nodes*31.98

            if current_cell.is_the_last_cell():
                current_cell_index = 0
            else:
                current_cell_index += 1

            asn += 1

        return success, fail, delays, total_transmissions, 0, energy, idle, sink_received, duplicated_packets,total_rx
