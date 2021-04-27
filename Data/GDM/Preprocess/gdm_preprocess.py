import pandas as pd
import numpy as np
from Mother import *
from MotherCollection import *
import networkx as nx
from Bacteria import *
import os
from openpyxl import load_workbook
import pickle


def create_tax_tree(series, zeroflag=False):
    tempGraph = nx.Graph()
    """workbook = load_workbook(filename="random_Otus.xlsx")
    sheet = workbook.active"""
    valdict = {}
    bac = []
    for i, (tax, val) in enumerate(series.items()):
        # adding the bacteria in every column
        bac.append(Bacteria(tax, val))
        # connecting to the root of the tempGraph
        tempGraph.add_edge("anaerobe", bac[i].lst[0])
        # connecting all levels of the taxonomy
        for j in range(0, len(bac[i].lst) - 1):
            updateval(tempGraph, bac[i], valdict, j, True)
        # adding the value of the last node in the chain
        updateval(tempGraph, bac[i], valdict, len(bac[i].lst) - 1, False)
    valdict["anaerobe"] = valdict["Bacteria"] + valdict["Archaea"]
    return create_final_graph(tempGraph, valdict, zeroflag)


def updateval(graph, bac, vald, num, adde):
    if adde:
        graph.add_edge(bac.lst[num], bac.lst[num + 1])
    # adding the value of the nodes
    if bac.lst[num] in vald:
        vald[bac.lst[num]] += bac.val
    else:
        vald[bac.lst[num]] = bac.val


def create_final_graph(tempGraph, valdict, zeroflag):
    graph = nx.Graph()
    for e in tempGraph.edges():
        if not zeroflag or valdict[e[0]] * valdict[e[1]] != 0:
            graph.add_edge((e[0], valdict[e[0]]),
                           (e[1], valdict[e[1]]))
    return graph


def iterate_dataframe(all_moms_dataframe, mapping_table_dataframe, save_to_pickle=False):
    mom_list = MotherCollection()
    taxonomy_series = all_moms_dataframe.iloc[-1]
    for index, mom in mapping_table_dataframe.iterrows():
        print("index", index)
        split_sample_id = mom['#SampleID'].split('-')
        id = split_sample_id[0]
        test_way = split_sample_id[1]
        trimester = split_sample_id[2][-1]
        repetition = split_sample_id[-1]
        sick_or_not = mom['Control_GDM']
        data = []
        ind = []
        for i in range(all_moms_dataframe.shape[1]):
            microbiome_id = all_moms_dataframe.columns[i]
            ind.append(taxonomy_series[microbiome_id])
            data.append(all_moms_dataframe.iloc[index - 1][microbiome_id])
        my_series = pd.Series(data, index=ind, dtype=int)
        microbiome_graph = create_tax_tree(my_series, zeroflag=True)
        mom_object = Mother(id, trimester, repetition, sick_or_not, microbiome_graph, test_way=test_way)
        mom_list.add_mom(mom_object)
    if save_to_pickle:
        pickle.dump(mom_list, open("mom_collection.p", "wb"))
    return mom_list


def find_joint_nodes_set(mom_collection):
    s = set()
    for mom in mom_collection.mother_list:
        cur_graph = mom.microbiome_graph
        nodes = cur_graph.nodes(data=False)
        for name, value in nodes:
            s.add(name)
    print("number of microbioms:", len(s))

def create_graph_files_for_qgcn(mom_list, file_name_save):
    f = open(file_name_save, "wt")
    for i, mom in enumerate(mom_list.mother_list):
        cur_graph = mom.microbiome_graph
        nodes = cur_graph.nodes(data=True)
        edges = cur_graph.edges(data=False)
        for u, v in cur_graph.edges(data=False):
            name1, value1 = u
            name2, value2 = v

    pass


def load_taxonomy_file(file_name):
    all_moms_dataframe = pd.read_csv(file_name, delimiter='\t', header=1)
    all_moms_dataframe = all_moms_dataframe.T
    new_header = all_moms_dataframe.iloc[0]  # grab the first row for the header
    all_moms_dataframe = all_moms_dataframe[1:]  # take the data less the header row
    all_moms_dataframe.columns = new_header  # set the header row as the df header
    all_moms_dataframe.index.name = 'ID'
    return all_moms_dataframe


def load_mom_details_file(file_name):
    wb = load_workbook(file_name)
    ws = wb['Sheet1']
    data = []
    for row in ws:
        if not ws.row_dimensions[row[0].row].hidden:
            row_values = [cell.value for cell in row]
            data.append(row_values)
    mapping_table_dataframe = pd.DataFrame(data[1:], columns=data[0])
    # mapping_table_dataframe = pd.read_excel(mapping_table, header=0)
    mapping_table_dataframe = mapping_table_dataframe[1:]
    mapping_table_dataframe.dropna(axis=1, inplace=True)
    mapping_table_dataframe.rename(columns={"#SampleID": "ID"}, inplace=True)

    return mapping_table_dataframe


if __name__ == '__main__':
    path_data_dir = os.path.join("..", "israel")
    taxonomy_file_name = os.path.join(path_data_dir, "israel_stool_otu_with_taxonomy.tsv")
    mapping_table = os.path.join(path_data_dir, "israeli_stool_mapping_table.xlsx")

    all_moms_dataframe = load_taxonomy_file(taxonomy_file_name)
    all_moms_dataframe.to_csv("taxonomy_for_mip_mlp.csv", index=False)
    mapping_table_dataframe = load_mom_details_file(mapping_table)
    mapping_table_dataframe.to_csv("tag_for_mip_mlp.csv", index=False)
    # iterate_dataframe(all_moms_dataframe, mapping_table_dataframe, save_to_pickle=True)

    # mom_list = pickle.load(open("mom_collection.p", "rb"))
    # class point():
    #     def __init__(self, x, y):
    #         self.x = x
    #         self.y = y
    #
    # G = nx.Graph()
    # p0 = point(0, 0)
    # p1 = point(1, 1)
    #
    # G.add_node(0, data=p0)
    # G.add_node(1, data=p1)
    # G.add_edge(0, 1, weight=4)
    # create_graph_files_for_qgcn(mom_list, "microbiome_data_for_qgcn")
    # find_joint_nodes_set(mom_list)
    print()
