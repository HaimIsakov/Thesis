import pandas as pd
import os


def merge_file(file1, file2):
    df1 = pd.read_excel(file1, header=0)
    df1.dropna(axis=1, inplace=True)
    df2 = pd.read_csv(file2, sep='\t', header=1).T

    new_header = df2.iloc[0]  # grab the first row for the header
    df2 = df2[1:]  # take the data less the header row
    df2.columns = new_header  # set the header row as the df header
    df2.index.name = 'ID'

    merged_df = pd.merge(df1, df2, left_on='#SampleID', right_on='ID', how='inner')
    merged_df.to_csv("merged_gdm_data.csv", index=False)


if __name__ == '__main__':
    path_data_dir = os.path.join("..", "israel")
    taxonomy_file_name = os.path.join(path_data_dir, "israel_stool_otu_with_taxonomy.tsv")
    mapping_table = os.path.join(path_data_dir, "israeli_stool_mapping_table.xlsx")

    merge_file(mapping_table, taxonomy_file_name)
