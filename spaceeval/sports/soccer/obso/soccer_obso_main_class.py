import os
from .get_obso_value import calculate_obso_fc
from .vis_obso import plot_pitchcontrol_for_event
import pandas as pd
import numpy as np
from tqdm import tqdm


class soccer_obso:
    def __init__(self, event_data = None, tracking_home = None, tracking_away = None, out_path = None, testing_mode = False):
        self.event_data = event_data
        self.tracking_home = tracking_home
        self.tracking_away = tracking_away
        self.testing_mode = testing_mode
        self.out_path = out_path

    def read_data(self):
        #check if self.event is a folder or a file
        if os.path.isdir(self.event_data):
            #read all files in the directory
            event_files = [os.path.join(self.event_data, f) for f in os.listdir(self.event_data) if f.endswith('.csv')]
            event_dict = {os.path.basename(f).split('.')[0].split('_')[-1]: pd.read_csv(f) for f in event_files}
        else:
            event_dict = {os.path.basename(self.event_data).split('.')[0].split('_')[-1]: pd.read_csv(self.event_data)}

        if os.path.isdir(self.tracking_home):
            tracking_home_files = [os.path.join(self.tracking_home, f) for f in os.listdir(self.tracking_home) if f.endswith('.csv')]
            tracking_home_dict = {os.path.basename(f).split('.')[0].split('_')[-1]: pd.read_csv(f) for f in tracking_home_files}
        else:
            tracking_home_dict = {os.path.basename(self.tracking_home).split('.')[0].split('_')[-1]: pd.read_csv(self.tracking_home)}

        if os.path.isdir(self.tracking_away):
            tracking_away_files = [os.path.join(self.tracking_away, f) for f in os.listdir(self.tracking_away) if f.endswith('.csv')]
            tracking_away_dict = {os.path.basename(f).split('.')[0].split('_')[-1]: pd.read_csv(f) for f in tracking_away_files}
        else:
            tracking_away_dict = {os.path.basename(self.tracking_away).split('.')[0].split('_')[-1]: pd.read_csv(self.tracking_away)}

        return event_dict, tracking_home_dict, tracking_away_dict

    def get_obso(self):
        event_dict, tracking_home_dict, tracking_away_dict = self.read_data()
        results = {}
        if self.testing_mode:
                print(f"In testing mode, only limited result will be calculated")
        for key in tqdm(event_dict.keys(), desc="Calculating OBSO for all matches"):
            if key in tracking_home_dict and key in tracking_away_dict:
                if self.testing_mode:
                    results[key] = calculate_obso_fc(event_dict[key].head(10), tracking_home_dict[key], tracking_away_dict[key])
                else:
                    results[key] = calculate_obso_fc(event_dict[key], tracking_home_dict[key], tracking_away_dict[key])
            else:
                print(f"Tracking data for {key} not found in home or away datasets.")
        if self.out_path:
            os.makedirs(self.out_path+'/'+'obso', exist_ok=True)
            for key in results.keys():
                results[key][0].to_pickle(self.out_path+'/'+'obso'+'/'+f'{key}_home_obso.pkl')
                results[key][1].to_pickle(self.out_path+'/'+'obso'+'/'+f'{key}_away_obso.pkl')
                results[key][2].to_pickle(self.out_path+'/'+'obso'+'/'+f'{key}_home_onball_obso.pkl')
                results[key][3].to_pickle(self.out_path+'/'+'obso'+'/'+f'{key}_away_onball_obso.pkl')
                np.save(self.out_path+'/'+'obso'+'/'+f'{key}_PPCF_dict.npy', results[key][4])

        return results

    def vis_obso(self, event_id, events_data, tracking_home, tracking_away, PPCF,out_path):
        #read event_data, tracking_home, tracking_away if it is a path
        if not isinstance(events_data, pd.DataFrame):
            events_data = pd.read_csv(events_data)
        if not isinstance(tracking_home, pd.DataFrame):
            tracking_home = pd.read_csv(tracking_home)
        if not isinstance(tracking_away, pd.DataFrame):
            tracking_away = pd.read_csv(tracking_away)
        if not isinstance(PPCF, dict):
            PPCF = np.load(PPCF, allow_pickle=True).item()

        fig, ax = plot_pitchcontrol_for_event(event_id, events_data, tracking_home, tracking_away, PPCF[event_id])
        if self.out_path:
            os.makedirs(self.out_path+"/vis_obso", exist_ok=True)
            fig.savefig(os.path.join(self.out_path, "vis_obso", f'obso_event_{event_id}.png'))
        elif out_path:
            os.makedirs(out_path+"/vis_obso", exist_ok=True)
            fig.savefig(os.path.join(out_path, "vis_obso", f'obso_event_{event_id}.png'))
        return fig, ax
