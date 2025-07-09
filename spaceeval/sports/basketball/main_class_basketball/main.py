from ..models.BIMOS import BIMOS
from ..models.BMOS import BMOS
import os
import tqdm
import json
from ..application.heatmap import plot_heat_map_frame, plot_heat_map_sequence


class space_model_basketball:
    def __init__(self, model_name):
        """
        Initializes the class with model name and ensures the required file is downloaded.

        :param model_name: str, the name of the model (e.g., "BIMOS", "BMOS")
        :param dest_path: str, the local path where the file will be saved
        """
        self.model_name = model_name

    def plot_heat_map_frame(self, data, save_path_folder, *args, **kwargs):

        if self.model_name == "BIMOS":
            attValues = BIMOS(data).values
        
        if self.model_name == "BMOS":
            attValues = BMOS(data).values

        plot_heat_map_frame(data, save_path_folder, attValues, *args, **kwargs)
        

    def plot_heat_map_sequence(self, data, save_path_folder,*args, **kwargs):

        if self.model_name == "BIMOS":
            model = "BIMOS"
        
        if self.model_name == "BMOS":
            model = "BMOS"

        plot_heat_map_sequence(model, data, save_path_folder,*args, **kwargs)
    

    def get_values(self, data, json_path=None):

        if self.model_name == "BIMOS":
            df_len = len(data)
            print(f"Number of rows in the DataFrame: {df_len}")
            result_dict = {}
            for i in tqdm.tqdm(range(df_len)):
                row = data.iloc[i]
                key = f"{row['game']}_{row['attackid']}_{row['f_id']}"
                value = BIMOS(data.iloc[[i]]).values.tolist()
                result_dict[key] = value
            
            if json_path:
                filename = "BIMOS_values.json"
                full_path = os.path.join(json_path, filename)
                
                os.makedirs(json_path, exist_ok=True)
                
                with open(full_path, 'w') as f:
                    json.dump(result_dict, f)
            return result_dict
        
        if self.model_name == "BMOS":
            df_len = len(data)
            print(f"Number of rows in the DataFrame: {df_len}")
            result_dict = {}
            for i in tqdm.tqdm(range(df_len)):
                row = data.iloc[i]
                key = f"{row['game']}_{row['attackid']}_{row['f_id']}"
                value = BMOS(data.iloc[[i]]).values.tolist()
                result_dict[key] = value
            
            if json_path:
                filename = "BMOS_values.json"
                full_path = os.path.join(json_path, filename)
                
                os.makedirs(json_path, exist_ok=True)
                
                with open(full_path, 'w') as f:
                    json.dump(result_dict, f)
            return result_dict

