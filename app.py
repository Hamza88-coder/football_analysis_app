import streamlit as st
import matplotlib.pyplot as plt
import os

from PIL import Image
from utils import read_video, save_video
import plotly.express as px
import plotly.graph_objects as go

from trackers import Tracker
from tacticDeep import TacticDeepchat
from heatMap import HeatMap
import tempfile
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_mouvement_estimator import CameraMovementEstimator
import numpy as np
from view_transformer import ViewTransformer
from speed_distance_estimator import SpeedAndDistance_Estimator
from penalty import FootballArmDetection
from utils import get_center_of_bbox, measure_distance
from ultralytics import YOLO

def main():
    from constants import point,list_coord_off,list_offside,point2
    new_tracks_penalty, output_video_frames_penalty=dict(),None
    length_vid=0
    
    st.set_page_config(page_title="Football Tactical Analysis App", layout="wide")
    st.title("Football Tactical Analysis App")
  

    # Sidebar
    st.sidebar.title("Options")

    video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov"],key="tactical analysis")
    
    # Sidebar
    st.sidebar.title("Check a penalty")
    video_file_penalty = st.sidebar.file_uploader("Upload a penalty video file", type=["mp4", "mov"],key="check the penalty")
    player_id_input = st.sidebar.text_input("Enter player ID for heatmap", "")
    show_heatmap = st.sidebar.checkbox("Show Heatmap")

     # Camera movement option
    camera_movement_option = st.sidebar.radio("Is the camera moving?", ("No", "Yes"), index=0)
    the_var = st.sidebar.radio("Is penalty?", ("yes", "no"), index=0)

    col1, col2 = st.columns([1, 1])  # Ajustez les ratios selon vos besoins

    if video_file_penalty is not None :
      
    # Assurez-vous que le répertoire temporaire existe
      temp_dir = "temp"
      if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
      
    
    # Enregistrez le fichier téléchargé dans le répertoire temporaire
      temp_file_path_penalty = os.path.join(temp_dir, video_file_penalty.name)
      with open(temp_file_path_penalty, "wb") as f:
        f.write(video_file_penalty.read())
    
    # Lisez les frames de la vidéo
      video_frames = read_video(temp_file_path_penalty)
      length_vid=len(video_frames)
      st.sidebar.text(f"Video loaded: {video_file_penalty.name}")
    
    # Traitez la vidéo
      new_tracks_penalty, output_video_frames_penalty,_ = process_video(video_frames,None,None,None,None)
      
    
   
      
    
      with col2 :
        # Enregistrez et affichez la vidéo traitée
        output_file_path_pen = os.path.join("output_videos", 'output_pen.mp4')
        save_video(output_video_frames_penalty, output_file_path_pen)
        output_file_path_pen = os.path.join("output_videos", 'output_pen_h264.mp4')
        st.video(output_file_path_pen)

      
        
        
    
    
     
       
    if video_file_penalty:  
         with col1:
        # Tracez le graphique en utilisant la valeur de la barre de progression
          plot_penalty(new_tracks_penalty, output_video_frames_penalty)
          st.title("Examen des Incidents de Pénalité et des Touchers de Main à Travers des Images")

    # Path to the directory containing the images
          image_dir = "image"

    # Load images
          images = load_images(image_dir)

          if len(images) == 0:
           st.write("No images found in the specified directory.")
           return

    # Display images in Streamlit
         st.write("Displaying Images:")
    
    # Using columns to display images in a grid
         cols = st.columns(4)  # Adjust the number of columns as needed

         for i, img in enumerate(images[:8]):  # Limit to 8 images
          with cols[i % 4]:
            st.image(img, caption=f"Image {i+1}")
    
       
      
        
        
    

    if video_file is not None:
        
        # Ensure the temporary directory exists
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save uploaded file to a temporary location
        temp_file_path = os.path.join(temp_dir, video_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(video_file.read())

        # Read video frames
        video_frames = read_video(temp_file_path)
        st.sidebar.text(f"Video loaded: {video_file.name}")
        if camera_movement_option == "Yes":
            select_rectangle(video_frames[0])
    
        # Process video
        tracks,output_video_frmaes,output_video_frames_tactic = process_video(video_frames,list_offside,point,point2,list_coord_off)

        # Save video
        output_file_path = os.path.join("output_videos", 'output_ta9_video_264_h264_h264.mp4')
        #save_video(output_video_frmaes, output_file_path)
        

        #save video tactic map
        # Save video
        
        
        output_file_path_tactic = os.path.join("output_videos", 'output_ta9_video_tactic_h264_h264.mp4')
        
        
        #save_video(output_video_frames_tactic, output_file_path_tactic)


        # Display output
        st.video(output_file_path)
        #list_offside,point,_=get_offside(tracks,output_video_frmaes)
        
        
        
        
            # Display video in sidebar
        with open(output_file_path_tactic, 'rb') as f:
            video_bytes = f.read()
            st.sidebar.video(video_bytes)
        if player_id_input and show_heatmap:
            player_id = int(player_id_input)
            display_heatmap(tracks, player_id)
        

        # Clean up temporary file
        os.remove(temp_file_path)

def process_video(video_frames,list_offside,point,point2,list_coord_off):
    on = False

    # Initialize Tracker
    tracker = Tracker('model/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Assign team colors
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Initialize the tactic model
    tactic = TacticDeepchat('tactic_deep/best_points.pt', tracks)
    new_tracks,output_video_frames_tactic= tactic.map_players(video_frames, tracks, read_from_stub=True, stub_path='newStubs/track_stubs.pkl')

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign ball to players
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control)>0:
             team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Camera movement estimator
    if on:
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

 

    # Speed and distance estimator
    speed_est = SpeedAndDistance_Estimator()
    new_tracks=speed_est.add_speed_and_distance_to_tracks(new_tracks)
    if list_offside is not None and list_coord_off is not None:
      output_video_frames = tracker.draw_annotations(video_frames, tracks,list_offside,point,point2,list_coord_off,team_ball_control=None)
    else:
       output_video_frames = tracker.draw_annotations(video_frames, tracks,None,None,None,None,team_ball_control=None)
       output_video_frames=speed_est.draw_speed_and_distance(output_video_frames,new_tracks)


    
    return tracks,output_video_frames,output_video_frames_tactic
def display_heatmap(new_tracks, player_id):
    # Create heatmap
    heatmap = HeatMap(new_tracks["players"])
    heatmap.extract_player_positions(player_id)
    fig = heatmap.create_heatmap(field_dims=(8, 6), bin_size=5)
    st.sidebar.pyplot(fig)
def plot_penalty(tracks,video_frames):
 
    arm_detector = FootballArmDetection()
    landmarks_list=[]
    image_height=0
    image_width=0
    for frame_num,player_track in enumerate(tracks["players"]):
            
                  
                if tracks["ball"][frame_num] is not None:
                 if 1 in tracks["ball"][frame_num]:  # Check if key 1 exists
                
                   ball_bbox=tracks["ball"][frame_num][1]['bbox']
                   
                   for track_id,track_info in player_track.items():
                     if track_id==222 and ball_bbox is not None:
                      bbox=track_info["bbox"]
                      
                      arm_landmarks,image_height, image_width= arm_detector.get_arm_player(video_frames[frame_num], bbox)
                      

                      if arm_landmarks is not None:
                           
                           landmarks_list.append(arm_landmarks)
   
                           
    right_shoulder_list = [(d['right_shoulder'].x*  image_width, d['right_shoulder'].y*image_height) for d in landmarks_list]
    left_shoulder_list = [(d['left_shoulder'].x*  image_width, d['left_shoulder'].y*image_height) for d in landmarks_list]
    right_elbow_list = [(d['right_elbow'].x*image_width, d['right_elbow'].y*image_height) for d in landmarks_list]
    left_elbow_list = [(d['left_elbow'].x*image_width, d['left_elbow'].y*image_height) for d in landmarks_list]
    right_wrist_list = [(d['right_wrist'].x*image_width, d['right_wrist'].y*image_height) for d in landmarks_list]
    left_wrist_list = [(d['left_wrist'].x*image_width, d['left_wrist'].y*image_height) for d in landmarks_list]

    fig, ax = plt.subplots(figsize=(8, 6))
    arm_detector.plot_ball_trajectory_and_landmarks_second(ax, tracks["ball"], right_shoulder_list, left_shoulder_list, right_elbow_list, left_elbow_list, right_wrist_list, left_wrist_list)
    st.pyplot(fig)
    
    


def select_rectangle(frame):
    camera_mouvement_list=[]


    st.write("Select a rectangle by clicking and dragging on the image.")
    
    # Display the frame
    fig = px.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fig.update_layout(dragmode="select", newshape=dict(line_color="cyan"))
    selected_data = st.plotly_chart(fig, use_container_width=True)
   

    # Entrée des coordonnées par l'utilisateur
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("Coordonnée x0", min_value=0, max_value=frame.shape[1], value=0)
        y0 = st.number_input("Coordonnée y0", min_value=0, max_value=frame.shape[0], value=0)
        camera_mouvement_list.append([x0,y0])
    with col2:
        x1 = st.number_input("Coordonnée x1", min_value=0, max_value=frame.shape[1], value=frame.shape[1])
        y1 = st.number_input("Coordonnée y1", min_value=0, max_value=frame.shape[0], value=frame.shape[0])
        camera_mouvement_list.append([x1,y1])
    # Dessiner le rectangle sur l'image
    if st.button("Afficher le rectangle"):
        rectangle_img = cv2.rectangle(frame.copy(), (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
        
        # Convertir l'image avec le rectangle en RGB pour Streamlit
        rectangle_img_rgb = cv2.cvtColor(rectangle_img, cv2.COLOR_BGR2RGB)
        
        # Dessiner les coordonnées sur l'image
        cv2.putText(rectangle_img_rgb, f"({x0}, {y0})", (int(x0), int(y0)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(rectangle_img_rgb, f"({x1}, {y1})", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Afficher l'image avec le rectangle et les coordonnées
        st.image(rectangle_img_rgb, caption='Image avec rectangle défini et coordonnées')

        # Enregistrer l'image modifiée avec le rectangle
        cv2.imwrite('rectangle_defined.png', rectangle_img)
    return camera_mouvement_list

import numpy as np


   
def load_images(image_dir):
    images = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(image_dir, file_name))
            images.append(img)
    return images


   
def get_offside(tracks,video_frames):
    list_frame_offside=[]
    point=[]
    list_coord_off=[]
    point2=[]
    bbox_barca,bbox_ronalo,bbox_barca2=None,None,None
 
    for frame_num,player_track in enumerate(tracks["players"]):
                
            
                  
                if tracks["ball"][frame_num] is not None:
                 
                 if 1 in tracks["ball"][frame_num]:  # Check if key 1 exists
                 
                
                   ball_bbox=tracks["ball"][frame_num][1]['bbox']
                   
                   
                   for track_id,track_info in player_track.items():
                     
                     if track_id==744  :
                      bbox_ronalo=track_info["bbox"]
                     if track_id==199:
                      bbox_barca=track_info["bbox"]
                     if track_id==515:
                        bbox_barca2=track_info["bbox"]
                     if bbox_barca is not None and bbox_ronalo is not None:
                      
                      if get_center_of_bbox(bbox_ronalo)[1]-get_center_of_bbox(bbox_barca)[1]>=0:
                         if get_center_of_bbox(bbox_barca)[1]-get_center_of_bbox(ball_bbox)[1]>=0:
                            print("fadwa")
                            list_coord_off.append(get_center_of_bbox(bbox_ronalo))
                            point.append(get_center_of_bbox(bbox_barca))
                            point2.append(get_center_of_bbox(bbox_barca2))
                            list_frame_offside.append(frame_num)

                            
    
    
    # Writing the output to constants.py
    with open('constants.py', 'a') as f:
        f.write(f"point2 = {point2}\n")
        f.write(f"point = {point}\n")
        f.write(f"list_offside = {list_frame_offside}\n")
        f.write(f"list_coord_off = {list_coord_off}\n")

    return list_frame_offside,point,list_coord_off
    





















if __name__ == '__main__':
    main()
