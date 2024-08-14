from ultralytics import YOLO
import supervision as sv
import pickle
import os
import pandas as pd
import numpy as np
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox,get_bbox_width,get_foot_position
class Tracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()

    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def is_handball(arm_landmarks, ball_bbox):
     if not ball_bbox:
        return False
    
     ball_x1, ball_y1, ball_x2, ball_y2 = ball_bbox
     for key, (x, y) in arm_landmarks.items():
        if ball_x1 <= x <= ball_x2 and ball_y1 <= y <= ball_y2:
            return True
     return False


    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self,frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
          #0.1 signifie que seules les détections avec une confiance d'au moins 10% seront retournées
          detections_batch=self.model.predict(frames[i:i+batch_size],conf=0.1)
          detections+=detections_batch
        return detections
    def get_object_tracks(self,frames,read_from_stub=False,stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
             with open(stub_path,'rb') as f:
                 tracks=pickle.load(f)
             return tracks

        detections=self.detect_frames(frames)
        tracks={"players":[],"referees":[],"ball":[]}

        for frame_num,detection in enumerate(detections):
            cls_names=detection.names
            cls_names_inv={v:k for k,v in cls_names.items()}
            #convert to supervision Detection format
            detection_supervision=sv.Detections.from_ultralytics(detection)
            #convert goalkeeper to a player
            for object_ind,class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id]=="goalkeeper":
                    detection_supervision.class_id[object_ind]=cls_names_inv["player"]
            #Track object
            detections_with_tracks=self.tracker.update_with_detections(detection_supervision)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detections_with_tracks:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                track_id=frame_detection[4]

                if cls_id==cls_names_inv['player']:
                    tracks["players"][frame_num][track_id]={"bbox":bbox}
                if cls_id==cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id]={"bbox":bbox}
            for frame_detection in detection_supervision:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                if cls_id==cls_names_inv['ball']:
                    tracks["ball"][frame_num][1]={"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
        return tracks
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    def draw_ligne_offside(self, frame, point1, point2):
        c1,c2= Tracker.find_right_triangle_points_with_hypotenuse(point1,point2)

        
        cv2.line(frame, (int(point1[0]), int(point1[1])),(int(c2[0]), int(c2[1])), (0, 255, 0), 1)
        cv2.line(frame, (int(point2[0]), int(point2[1])),  (int(c1[0]), int(c1[1])), (0, 255, 0), 1)


        return frame
    
      
        
    def draw_traingle(self,frame,bbox,color):
        
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    def draw_traingle_offside(self,frame,point,color):
        
        y= int(point[1])
        x = point[0]
        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    def draw_rectangle(self, frame,point, color=(0, 0, 255), thickness=2, text="offside"):
        top_left=(point[0]-30,point[1]-30)
        bottom_right=(point[0]+50,point[1]+50)
       
        # Dessine le rectangle sur l'image
        cv2.rectangle(frame, top_left, bottom_right, color, thickness)
        
        # Définir la position du texte (au-dessus du rectangle)
        text_position = (top_left[0], top_left[1] - 10)
        
        # Ajouter le texte à l'image
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
         # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
    def find_right_triangle_points_with_hypotenuse(A, B):
      x1, y1 = A
      x2, y2 = B
    
    # Calcul du milieu de AB
      midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    # Calcul de la distance AB (diamètre du cercle)
      radius = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2
    
    # Points C au-dessus et en dessous de AB pour un angle droit
      C1 = (midpoint[0] + radius * np.cos(np.pi / 2), midpoint[1] + radius * np.sin(np.pi / 2))
      C2 = (midpoint[0] + radius * np.cos(-np.pi / 2), midpoint[1] + radius * np.sin(-np.pi / 2))
    
    # Retourner les coordonnées de C1 et C2
      return C1, C2

    def draw_annotations(self,video_frames,tracks,list_offside,point,point2,list_coord_off,team_ball_control):
        output_video_frames=[]
        i=0
        for frame_num,frame in enumerate (video_frames):
            
            frame=frame.copy()
            player_dict=tracks["players"][frame_num]
            ball_dict=tracks["ball"][frame_num]
            referee_dict=tracks["referees"][frame_num]
            #draw players
            for track_id,player in player_dict.items():
                color=player.get("team_color",(0,0,255))
                frame=self.draw_ellipse(frame,player["bbox"],color,track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(255,0,0))


            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))



            #draw Team Ball Control
            if team_ball_control is not None:
             frame=self.draw_team_ball_control(frame,frame_num,team_ball_control)
            if list_offside is not None and frame_num in list_offside:
                
                
                frame=self.draw_traingle_offside(frame,list_coord_off[i],(0,0,255))
                frame=self.draw_rectangle(frame,list_coord_off[i])
                i=i+1

            output_video_frames.append(frame)
        return output_video_frames

           



            



