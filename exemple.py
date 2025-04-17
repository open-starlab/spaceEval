from preprocessing import Space_data
from spaceeval import Space_Model

#download the data
# indicate the folder with the sportVU data (Rikako Kono shape)
data_path = "data_VUNBA"

Space_data(data_provider="SportVU_NBA", data_path= data_path).download_data()

basket_df = Space_data(data_provider="SportVU_NBA", data_path= data_path).preprocessing(nb_process_game = 4)

# Initialize the sport and the model of space evaluation you want
space_model = Space_Model('BIMOS')

# Create a DataFrame with just row 100 we want visualize
data_frame = basket_df.iloc[[100]].copy()

#chose a save path folder
save_path_folder = "C:/Users/titou/Downloads"

# Plot the heat map 
space_model.plot_heat_map_frame(save_path_folder, data_frame)