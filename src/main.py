
from modules.data_loader import DataLoader
from evaluators.evaluator import Evaluator
import os
import argparse
import pandas as pd

##############################################################
# Argparser
##############################################################
def define_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--src_folder_path', type=str, default="/workspace/company/math_scoring_with_gpt/data")
    parser.add_argument('--dst_folder_path', type=str, default="/workspace/company/math_scoring_with_gpt/results")

    cfg = parser.parse_args()

    return cfg

def save_result(df, file_name):
    ##############################################################
    # Save
    ##############################################################
    
    save_path = os.path.join(
            cfg.dst_folder_path, 
            file_name
        )
    
    df.to_csv(
        save_path,
        sep="\t", 
        index=False
        )
    print("Saved: ", save_path)

##############################################################
# Main Function
##############################################################
def main(cfg):

    ##############################################################
    # Data loader
    ##############################################################
    data_loader = DataLoader(data_folder_path=cfg.src_folder_path)
    file_name_list = os.listdir(cfg.src_folder_path)
    file_name_list = [file_name for file_name in file_name_list if file_name.endswith(".tsv")]

    ##############################################################
    # Evalutator
    ##############################################################
    evaluator = Evaluator()

    ##############################################################
    # Loop over files
    ##############################################################
    for file_name in file_name_list:

        df = data_loader.load_data(file_name=file_name)

        if file_name == "number_1.tsv":
            eval_number_1_1_df = evaluator.eval_number_1_1(df=df)
            eval_number_1_1_df_file_name = file_name.replace(".tsv", "_1_eval.tsv")
            save_result(eval_number_1_1_df, eval_number_1_1_df_file_name)

            eval_number_1_2_df = evaluator.eval_number_1_2(df=df)
            eval_number_1_2_df_file_name = file_name.replace(".tsv", "_2_eval.tsv")
            save_result(eval_number_1_2_df, eval_number_1_2_df_file_name)

        elif file_name == "number_2.tsv":
            eval_number_2_1_df = evaluator.eval_number_2_1(df=df)
            eval_number_2_1_df_file_name = file_name.replace(".tsv", "_1_eval.tsv")
            save_result(eval_number_2_1_df, eval_number_2_1_df_file_name)

            eval_number_2_2_df = evaluator.eval_number_2_2(df=df)
            eval_number_2_2_df_file_name = file_name.replace(".tsv", "_2_eval.tsv")
            save_result(eval_number_2_2_df, eval_number_2_2_df_file_name)

            eval_number_2_3_df = evaluator.eval_number_2_3(df=df)
            eval_number_2_3_df_file_name = file_name.replace(".tsv", "_3_eval.tsv")
            save_result(eval_number_2_3_df, eval_number_2_3_df_file_name)

        elif file_name == "number_3.tsv":
            eval_number_3_1_df = evaluator.eval_number_3_1(df=df)
            eval_number_3_1_df_file_name = file_name.replace(".tsv", "_1_eval.tsv")
            save_result(eval_number_3_1_df, eval_number_3_1_df_file_name)

            eval_number_3_2_df = evaluator.eval_number_3_2(df=df)
            eval_number_3_2_df_file_name = file_name.replace(".tsv", "_2_eval.tsv")
            save_result(eval_number_3_2_df, eval_number_3_2_df_file_name)


        elif file_name == "number_4.tsv":
            eval_number_4_1_df = evaluator.eval_number_4_1(df=df)
            eval_number_4_1_df_file_name = file_name.replace(".tsv", "_1_eval.tsv")
            save_result(eval_number_4_1_df, eval_number_4_1_df_file_name)

            eval_number_4_2_df = evaluator.eval_number_4_2(df=df)
            eval_number_4_2_df_file_name = file_name.replace(".tsv", "_2_eval.tsv")
            save_result(eval_number_4_2_df, eval_number_4_2_df_file_name)

        elif file_name == "number_5.tsv":
            eval_number_5_1_df = evaluator.eval_number_5_1(df=df)
            eval_number_5_1_df_file_name = file_name.replace(".tsv", "_1_eval.tsv")
            save_result(eval_number_5_1_df, eval_number_5_1_df_file_name)

            eval_number_5_2_df = evaluator.eval_number_5_2(df=df)
            eval_number_5_2_df_file_name = file_name.replace(".tsv", "_2_eval.tsv")
            save_result(eval_number_5_2_df, eval_number_5_2_df_file_name)

        else:
            raise ValueError("File name is not valid!")

##############################################################
# Main Process
##############################################################
if __name__ == '__main__':

    cfg = define_argparser()

    main(cfg)