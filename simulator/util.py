#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Rodrigo Teles Hermeto

import logging
import sys
import psycopg2

logging.basicConfig(filename='experiment.log', level=logging.INFO, filemode='w', format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p')

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(message)s')
sh.setFormatter(formatter)
logging.getLogger().addHandler(sh)

def print_log(text, header=True):
    if header:
        logging.info("********************* {} *********************".format(text))
    else:
        logging.info("{}".format(text))
    logging.info("")


def get_database_connection():
    # change here to use your database
    return psycopg2.connect(database='your_database', user='your_user', password='your_password', host='your_host',port='port')


def get_all_nodes(experiment_id):
    query = "select distinct node from anycast_transmissions where experiment_id = {})  order by node".format(experiment_id)
    conn = get_database_connection()
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    nodes = []
    for row in rows:
        nodes.append(row[0])

    cur = conn.cursor()
    cur.execute("select sender from anycast_transmissions where experiment_id = {} limit 1".format(experiment_id))
    row = cur.fetchone()
    nodes.insert(0, row[0])
    conn.close()
    return nodes
