
import pandas as pd
import os


class DataLoader():
    def __init__(
            self, 
            data_folder_path
            ):
        super().__init__()

        self.data_folder_path = data_folder_path

    def load_data(self, file_name):

        final_path = os.path.join(self.data_folder_path, file_name)

        df = pd.read_csv(final_path, sep="\t")

        return df
