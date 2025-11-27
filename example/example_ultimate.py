# from preprocessing import Space_data
from spaceeval import Space_Model

event_path = "./UFA/EventData"  # path to the event data
tracking_path = "./UFA/TrackingData"  # path to the tracking data
out_path = "./output"  # path to save the processed data
processed_event_path = "./output/event"  # path relative to out_path where the processed event data is saved
home_tracking_path = "./output/home"  # path relative to out_path where the processed home tracking data is saved
away_tracking_path = "./output/away"  # path relative to out_path where the processed away tracking data is saved

# Space_data(data_provider='UFA',
#         event_data_path=event_path,
#         tracking_data_path=tracking_path,
#         out_path=out_path
#         ).preprocessing()

model = Space_Model(
    space_model="wUPPCF",
    event_data=processed_event_path,
    tracking_home=home_tracking_path,
    tracking_away=away_tracking_path,
    provider="UFA",
    out_path=out_path,
)

results = model.get_wuppcf()

model.vis_wuppcf(results)
