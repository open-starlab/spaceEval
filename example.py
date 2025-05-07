from preprocessing import Space_data
from spaceeval import Space_Model
import pandas as pd

#---- data preparation------

# if you need the preprocessing data

# indicate the folder where you gonna download the sportVU data (Rikako Kono shape)
data_path = "your/folder/data"

# download the data
Space_data(data_provider="SportVU_NBA", data_path= data_path).download_data()

#reshape the data from 4 game in CSV
basket_df = Space_data(data_provider="SportVU_NBA", data_path= data_path).preprocessing(nb_process_game = 4)


# if you already have the preprocessing data reshape_data.csv
basket_df = pd.read_csv("your/folder/data/reshape_data.csv")


#--------- space_eval -----------
# Initialize the sport and the model of space evaluation you want
space_model = Space_Model('BIMOS')

# Create a DataFrame with just row 100 we want visualize
data_frame = basket_df.iloc[[100]].copy()

#chose a save path folder to save the plot or the figure
save_path_folder = data_path

# Plot the heat map for frame
space_model.plot_heat_map_frame(save_path_folder, data_frame)


# Filter only the row for the event we want to visualize
data_frame = basket_df[(basket_df['game'] == 1) & (basket_df['eventid']==6)].copy()

# Create the video with heat map for event frame during sequence
space_model.plot_heat_map_sequence(data_frame,data_path)
