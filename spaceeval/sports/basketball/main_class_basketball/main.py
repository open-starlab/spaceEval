from ..models.BIMOS import BIMOS
from ..models.BMOS import BMOS
import os
import gdown
import zipfile

from ..application.heatmap import plot_heat_map_frame, plot_heat_map_sequence

class space_model_basketball:
    def __init__(self, model_name):
        """
        Initializes the class with model name and ensures the required file is downloaded.

        :param model_name: str, the name of the model (e.g., "BIMOS", "BMOS")
        :param file_id: str, the Google Drive file ID
        :param dest_path: str, the local path where the file will be saved
        """
        self.model_name = model_name

    def plot_heat_map_frame(self, save_path_folder, data, *args, **kwargs):

        if self.model_name == "BIMOS":
            attValues = BIMOS(data).values
        
        if self.model_name == "BMOS":
            attValues = BMOS(data).values

        plot_heat_map_frame(save_path_folder, attValues, data, *args, **kwargs)
        

    def plot_heat_map_sequence(self, data, save_path_folder,*args, **kwargs):

        if self.model_name == "BIMOS":
            model = "BIMOS"
        
        if self.model_name == "BMOS":
            model = "BMOS"

        plot_heat_map_sequence(model, data, save_path_folder,*args, **kwargs)

