import matplotlib.colors
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_center_of_bbox
import pandas as pd
import mplsoccer 

class HeatMap:
    def __init__(self, tracks_players):
        self.positions = []
        self.tracks_players = tracks_players
    
    def extract_player_positions(self, player_track_id):
        for frame_num, track in enumerate(self.tracks_players):
            for track_id, track_info in track.items():
                if track_id == player_track_id:
                    position = track_info['adjusted_position_chat']
                    
                    self.positions.append(position)
    
    def create_heatmap(self, field_dims, bin_size):
     
        positions_array = np.array(self.positions)
        x=positions_array[:, 0]
        x=list(x)
        y=positions_array[:, 1]
        y=list(y)
        x=self.normalize_to_range(x,0,100)
        y=self.normalize_to_range(y,0,100)
        
       
    
        pitch = mplsoccer.VerticalPitch(pitch_type='opta', pitch_color='black', line_color='white', line_zorder=2)
        fig, ax = pitch.draw(figsize=(8, 16))
        fig.set_facecolor('black')
        custommap=matplotlib.colors.LinearSegmentedColormap.from_list('custom map',['black','red'])
        pitch.kdeplot(x, y, ax=ax,cmap=custommap,shade=True,n_levels=100,zorder=1)
        return fig

    def normalize_to_range(self,lst, new_min=0, new_max=100):
      old_min = min(lst)
      old_max = max(lst)
    
    # Normalisation de chaque élément
      normalized_lst = [(new_max - new_min) * (x - old_min) / (old_max - old_min) + new_min for x in lst]
    
      return normalized_lst

    