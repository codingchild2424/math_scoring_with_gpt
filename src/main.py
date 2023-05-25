
from modules.data_loader import DataLoader
from evaluators.evaluator import Evaluator
import os
import argparse

##############################################################
# Argparser
##############################################################
def define_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--src_folder_path', type=str, default="/workspace/company/math_scoring_with_gpt/data")
    parser.add_argument('--dst_folder_path', type=str, default="/workspace/company/math_scoring_with_gpt/results")

    cfg = parser.parse_args()

    return cfg

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

        print("File name: ", file_name)

        df = data_loader.load_data(file_name=file_name)

        if file_name == "number_1.tsv":
            eval_df = evaluator.eval_number_1(df=df)
        # elif file_name == "number_2.tsv":
        #     eval_df = evaluator.eval_number_2(df=df)
        # elif file_name == "number_3.tsv":
        #     eval_df = evaluator.eval_number_3(df=df)
        # elif file_name == "number_4.tsv":
        #     eval_df = evaluator.eval_number_4(df=df)
        # elif file_name == "number_5.tsv":
        #     eval_df = evaluator.eval_number_5(df=df)
        else:
            raise ValueError("File name is not valid!")

        ##############################################################
        # Save
        ##############################################################
        eval_df.to_csv(
            os.path.join(
                cfg.dst_folder_path, 
                file_name.replace(".tsv", "_eval.tsv")
            ),
            sep="\t", 
            index=False
            )
        print("Saved: ", os.path.join(cfg.dst_folder_path, file_name.replace(".tsv", "_eval.tsv")))

##############################################################
# Main Process
##############################################################
if __name__ == '__main__':

    cfg = define_argparser()

    main(cfg)