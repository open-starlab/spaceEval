#from preprocessing import Space_data
from spaceeval import Space_Model
import pandas as pd

#---- space data -----

# indicate the folder with the sportVU data (Rikako Kono shape)
data_path = "data_VUNBA"

##for download the data
#Space_data(data_provider="SportVU_NBA", data_path= data_path).download_data()

##for reshape the data
#basket_df = Space_data(data_provider="SportVU_NBA", data_path= data_path).preprocessing(nb_process_game = 4)


#--------- space_eval -----------
basket_df = pd.read_csv("data_VUNBA/basket_df.csv")

# Initialize the sport and the model of space evaluation you want
space_model = Space_Model('BIMOS')

# Create a DataFrame with just row 100 we want visualize
data_frame = basket_df.iloc[[100]].copy()

#chose a save path folder
save_path_folder = data_path

# Plot the heat map for frame
space_model.plot_heat_map_frame(save_path_folder, data_frame)

# # Plot the heat map for sequence (very very long)
# # filter only the row for the event we want to visualize
# data_frame = basket_df[(basket_df['game_id'] == 1) & (basket_df['eventid']==6)].copy()

# space_model.plot_heat_map_sequence(data_frame)
