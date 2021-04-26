import pandas as pd
import numpy as np
from Mother import *
from MotherCollection import *

taxonomy_file_name = "israel_stool_otu_with_taxonomy.tsv"
mapping_table = "israeli_stool_mapping_table.xlsx"


def create_microbiome_graph_from_mother(mother):
    pass


def create_mother_details(mapping_table_file):
    mapping_table = pd.read_excel(mapping_table_file)
    split_sample_id = mapping_table['#SampleID'].split('-')
    id = split_sample_id[0]
    test_way = split_sample_id[1]
    trimester = split_sample_id[2][-1]
    repetition = split_sample_id[-1]
    sick_or_not = mapping_table['Control_GDM'].map({'Control': 0, 'GDM': 1})
    mother = Mother(id, trimester, repetition, sick_or_not, test_way)
    return mother


def create_mothers_list(taxonomy_file_name):
    mother_list = MotherCollection()
    # transpose
    all_moms_dataframe = pd.read_csv(taxonomy_file_name, delimiter='\t').T
    x = all_moms_dataframe.apply(create_microbiome_graph_from_mother)





