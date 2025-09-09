class Space_Model:
    basketball_space_model = ['BIMOS','BMOS']
    soccer_space_model = ['soccer_OBSO','soccer_BIMOS']
    other_model = []

    def __new__(cls, space_model, *args, **kwargs):
        if space_model in cls.basketball_space_model:
            from .basketball.main_class_basketball.main import space_model_basketball
            return space_model_basketball(space_model, *args, **kwargs)
        elif space_model in cls.soccer_space_model:
            from .soccer.soccer_main_func import space_model_soccer
            return space_model_soccer(space_model, *args, **kwargs)
        elif space_model in cls.other_model:
            raise NotImplementedError('other model not implemented yet')
        else:
            raise ValueError(f'Unknown event model: {space_model}')

if __name__ == '__main__':
    # import os
    # import pandas as pd
    # data = pd.read_csv('./starlab/all_data_reduce.csv')
    # # Filter data to keep only rows where game=1 and event_id=6
    # data_frame = data[(data['game'] == 1) & (data['eventid'] == 6)].copy()
    # # Reduce data_frame to only the first row
    # data_frame = data_frame.iloc[[1]].copy()
    # print(data_frame)

    # # Initialize the sport and the model of space evaluation you want
    # space_model = Space_Model('BIMOS')

    # #chose a save path folder
    # save_path_folder = "./Downloads"

    # #Ensure the save pat directory exists
    # os.makedirs(save_path_folder, exist_ok=True)

    # # Plot the heat map 
    # space_model.plot_heat_map_frame(save_path_folder, data_frame)

    print('Done')
