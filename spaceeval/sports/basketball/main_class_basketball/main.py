from ..models.BIMOS import BIMOS
from ..models.BMOS import BMOS
import os
import requests
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
        
        # Ensure the required file is downloaded
        self.download_file_if_needed()

    def download_file_from_gdrive(self):
        """
        Downloads and extracts the onball_scoreDataset ZIP file from Google Drive using gdown.
        """
        # === Configuration ===
        current_path = os.getcwd()
        self.dest_path = os.path.join(current_path, "onball_scoreDataset")
        
        zip_filename = "onball_scoreDataset.zip"
        zip_path = os.path.join(current_path, zip_filename)

        # Remplace par l'ID exact de ton fichier .zip dans Google Drive
        file_id = "14vSxQxTgRhuZAzqJTZLJgkiGDwbsmY1x"
        url = f"https://drive.google.com/uc?id={file_id}"

        print("Downloading ZIP archive from Google Drive...")

        try:
            # Téléchargement du fichier ZIP
            gdown.download(url, output=zip_path, quiet=False)

            print("Download completed. Extracting...")

            # Création du dossier si nécessaire
            if not os.path.exists(self.dest_path):
                os.makedirs(self.dest_path)

            # Extraction du fichier ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(current_path)
                
            # Clean up - remove the zip file after extraction
            os.remove(zip_path)

            print(f"✅ Extraction completed to folder: {self.dest_path}")

        except Exception as e:
            print(f"❌ Error during download or extraction: {e}")

    def download_file_if_needed(self):
        """
        Checks if the file is already downloaded. If not, downloads it.
        """
        if not os.path.exists("onball_scoreDataset"):
            self.download_file_from_gdrive()
        else:
            print("Necessary data already downloaded.")

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

