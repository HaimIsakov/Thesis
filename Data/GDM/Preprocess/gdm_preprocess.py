import pandas as pd
import numpy as np
from Mother import *
from MotherCollection import *
import networkx as nx
from Bacteria import *
import os


def create_tax_tree(series, zeroflag=False):
    tempGraph = nx.Graph()
    valdict = {}
    bac = []
    for i, (tax, val) in enumerate(series.items()):
        # adding the bacteria in every column
        bac.append(Bacteria(tax, val))
        # connecting to the root of the tempGraph
        tempGraph.add_edge("base", bac[i].lst[0])
        # connecting all levels of the taxonomy
        for j in range(0, len(bac[i].lst) - 1):
            updateval(tempGraph, bac[i], valdict, j, True)
        # adding the value of the last node in the chain
        updateval(tempGraph, bac[i], valdict, len(bac[i].lst) - 1, False)
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


def create_mother_details(mapping_table_dataframe):
    split_sample_id = mapping_table_dataframe['#SampleID'].split('-')
    id = split_sample_id[0]
    test_way = split_sample_id[1]
    trimester = split_sample_id[2][-1]
    repetition = split_sample_id[-1]
    sick_or_not = mapping_table_dataframe['Control_GDM'].map({'Control': 0, 'GDM': 1})
    microbiome_graph = create_tax_tree(series, zeroflag=False)
    mother = Mother(id, trimester, repetition, sick_or_not, microbiome_graph, test_way)
    return mother


def create_mothers_list(all_moms_dataframe):
    mother_list = MotherCollection()
    # transpose
    all_moms_dataframe = all_moms_dataframe.T
    # x = all_moms_dataframe.apply(create_microbiome_graph_from_mother)
    print()


def iterate_dataframe(all_moms_dataframe, mapping_table_dataframe):

    for index, mom in mapping_table_dataframe.iterrows():
        split_sample_id = mom['#SampleID'].split('-')
        id = split_sample_id[0]
        test_way = split_sample_id[1]
        trimester = split_sample_id[2][-1]
        repetition = split_sample_id[-1]
        sick_or_not = mom['Control_GDM']

        microbiome_graph = create_tax_tree(mom, zeroflag=False)
        mom_object = Mother(id, trimester, repetition, sick_or_not, microbiome_graph, test_way='stool')




if __name__ == '__main__':
    path_data_dir = os.path.join("..", "israel")
    taxonomy_file_name = os.path.join(path_data_dir, "israel_stool_otu_with_taxonomy.tsv")
    mapping_table = os.path.join(path_data_dir, "israeli_stool_mapping_table.xlsx")

    all_moms_dataframe = pd.read_csv(taxonomy_file_name, delimiter='\t', header=1)
    all_moms_dataframe = all_moms_dataframe.T
    new_header = all_moms_dataframe.iloc[0]  # grab the first row for the header
    all_moms_dataframe = all_moms_dataframe[1:]  # take the data less the header row
    all_moms_dataframe.columns = new_header  # set the header row as the df header
    all_moms_dataframe.index.name = 'ID'

    mapping_table_dataframe = pd.read_excel(mapping_table, header=0)
    mapping_table_dataframe = mapping_table_dataframe[1:]
    mapping_table_dataframe.dropna(axis=1, inplace=True)
    iterate_dataframe(all_moms_dataframe, mapping_table_dataframe)
