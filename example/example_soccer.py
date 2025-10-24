from preprocessing import Space_data
from spaceeval import Space_Model

event_path = './FIFA_WC_2022/Event Data' # path to the event data
tracking_path = './FIFA_WC_2022/Tracking Data' # path to the tracking data
out_path = './' # path to save the processed data
processed_event_path = './event'  #path relative to out_path where the processed event data is saved
home_tracking_path = './home_tracking' #path relative to out_path where the processed home tracking data is saved
away_tracking_path = './away_tracking' #path relative to out_path where the processed away tracking data is saved

Space_data(data_provider='fifa_wc_2022',
        event_data_path=event_path,
        tracking_data_path=tracking_path,
        out_path=out_path
        ).preprocessing()

model = Space_Model(space_model='soccer_OBSO',
            event_data=processed_event_path ,
            tracking_home=home_tracking_path,
            tracking_away=away_tracking_path,
            out_path=out_path)

model.get_obso()