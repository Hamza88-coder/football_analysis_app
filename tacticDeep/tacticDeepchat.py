# Import libraries
import numpy as np
import pandas as pd
import cv2
import pickle
import skimage
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error
import json
import yaml
import os
import time
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class TacticDeepchat:
    def __init__(self, model_path,tracks):
        self.model_keypoints = YOLO(model_path)
        self.tracks=tracks
        self.imagepath = 'tactic_deep/tactical map.jpg'
        # Initialize frame counter
        self.keypoints_displacement_mean_tol = 10
        self.yamlpath = "tactic_deep/config pitch dataset.yaml"
        self.frame_nbr = 0
        # Count consecutive frames with no ball detected
        self.nbr_frames_no_ball = 0
        # Threshold for number of frames with no ball to reset ball track (frames)
        self.nbr_frames_no_ball_thresh = 30
        # Distance threshold for ball tracking (pixels)
        self.ball_track_dist_thresh = 100
        # Maximum ball track length (detections)
        self.max_track_length = 35
        self.colors=[]
        json_path = "tactic_deep/pitch map labels position.json"
        with open(json_path, 'r') as f:
            self.keypoints_map_pos = json.load(f)

    def extract_keypoints(self, frame, conf=0.6):
        results_keypoints = list(self.model_keypoints(frame, conf))
        bboxes_k = results_keypoints[0].boxes.xyxy.cpu().numpy()  # Detected field keypoints (x,y,w,h) bounding boxes
        bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy()  # Detected field keypoints (x,y,w,h) bounding boxes
        labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy())
        return bboxes_k, bboxes_k_c, labels_k

    def map_players(self, video_frames, tracks,read_from_stub=False,stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
             with open(stub_path,'rb') as f:
                 tracks=pickle.load(f)
             return tracks,None
        ball_track_history = {'src': [], 'dst': []}
        output_video_frames = []

        for frame in video_frames  :
            
            
            bboxes_p_0,colors = self.extract_boxes_p(tracks["players"], self.frame_nbr)
            bboxes_p_2 = self.extract_boxes_b(tracks, self.frame_nbr)
            self.frame_nbr += 1
            tac_map = cv2.imread(self.imagepath)
            # Reset ball tracks
            if self.nbr_frames_no_ball > self.nbr_frames_no_ball_thresh:
                ball_track_history['dst'] = []
                ball_track_history['src'] = []
            bboxes_k, bboxes_k_c, labels_k = self.extract_keypoints(frame)
            # Convert detected numerical labels to alphabetical labels
            classes_names_dic = self.classes_nammes(self.yamlpath)
            detected_labels = [classes_names_dic[i] for i in labels_k]
            # Extract detected field keypoints coordinates on the current frame
            detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])
            # Get the detected field keypoints coordinates on the tactical map
            detected_labels_dst_pts = np.array([self.keypoints_map_pos[i] for i in detected_labels])
            detected_labels_prev = []
            detected_labels_src_pts_prev = []

            h, detected_labels_prev, detected_labels_src_pts_prev = self.homography_matrix(
                detected_labels, self.frame_nbr, detected_labels_prev, detected_labels_src_pts,
                detected_labels_src_pts_prev, detected_labels_dst_pts)

            if bboxes_p_0:
                bboxes_p_0 = np.array(bboxes_p_0)
                # Get coordinates of detected players on frame (x_center, y_center+h/2)
                detected_ppos_src_pts = bboxes_p_0[:, :2] + np.array([[0] * bboxes_p_0.shape[0], bboxes_p_0[:, 3] / 2]).transpose()
            else:
                detected_ppos_src_pts = np.array([])

            # Get coordinates of the first detected ball (x_center, y_center)
            detected_ball_src_pos = tracks["ball"][self.frame_nbr-1][1]['bbox'][:2] if tracks["ball"][self.frame_nbr-1] else None

            pred_dst_pts = []
           
           
            if detected_ppos_src_pts.size > 0 and h is not None:
               
                # Transform players coordinates from frame plane to tactical map plane using the calculated Homography matrix
                for pt in detected_ppos_src_pts:
                    pt_homogeneous = np.append(pt, 1)
                    dest_point = np.matmul(h, pt_homogeneous)
                    dest_point = dest_point / dest_point[2]
                    pred_dst_pts.append(dest_point[:2])
                pred_dst_pts = np.array(pred_dst_pts)
                self.ajout_adjusted_position(pred_dst_pts,self.frame_nbr-1)
                

            detected_ball_dst_pos = None
            
            if detected_ball_src_pos is not None and h is not None:
                pt_homogeneous = np.append(detected_ball_src_pos ,1)
                dest_point = np.matmul(h, pt_homogeneous)
                dest_point = dest_point / dest_point[2]
                detected_ball_dst_pos = dest_point[:2]
                self.ajout_adjusted_position_ball(detected_ball_dst_pos,self.frame_nbr-1)
                # Update track ball position history
                if len(ball_track_history['src']) > 0:
                    # Convertir detected_ball_src_pos en tableau NumPy si ce n'est pas déjà fait
                    detected_ball_src_pos = np.array(detected_ball_src_pos)

# Assurez-vous que ball_track_history['src'][-1] est également un tableau NumPy
                    ball_track_src_last = np.array(ball_track_history['src'][-1])
                    if np.linalg.norm(detected_ball_src_pos - ball_track_history['src'][-1]) < self.ball_track_dist_thresh:
                        ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                        ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                    else:
                        ball_track_history['src'] = [(int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))]
                        ball_track_history['dst'] = [(int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))]
                else:
                    ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                    ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))

            if len(ball_track_history['src']) > self.max_track_length:
                ball_track_history['src'].pop(0)
                ball_track_history['dst'].pop(0)

            annotated_frame = self.draw_on_tactic_map(tac_map, pred_dst_pts, detected_ball_dst_pos, ball_track_history,colors)
            output_video_frames.append(annotated_frame)
            if stub_path is not None:
             with open(stub_path,'wb') as f:
                pickle.dump(self.tracks,f)
        return self.tracks, output_video_frames
            

       

    def homography_matrix(self, detected_labels, frame_nbr, detected_labels_prev, detected_labels_src_pts,
                          detected_labels_src_pts_prev, detected_labels_dst_pts):
        h=None
        # Always calculate homography matrix on the first frame
        if frame_nbr > 1:
            # Determine common detected field keypoints between previous and current frames
            common_labels = set(detected_labels_prev) & set(detected_labels)
            # When at least 4 common keypoints are detected, determine if they are displaced on average beyond a certain tolerance level
            if len(common_labels) > 3:
                common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]  # Get labels indexes of common detected keypoints from previous frame
                common_label_idx_curr = [detected_labels.index(i) for i in common_labels]  # Get labels indexes of common detected keypoints from current frame
                coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]  # Get labels coordinates of common detected keypoints from previous frame
                coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]  # Get labels coordinates of common detected keypoints from current frame
                coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)  # Calculate error between previous and current common keypoints coordinates
                update_homography = coor_error > self.keypoints_displacement_mean_tol  # Check if error surpassed the predefined tolerance level
            else:
                update_homography = True
        else:
            update_homography = True

        if update_homography and len(detected_labels_src_pts)>=4:
           
            h, mask = cv2.findHomography(detected_labels_src_pts, detected_labels_dst_pts)

        detected_labels_prev = detected_labels.copy()  # Save current detected keypoint labels for next frame
        detected_labels_src_pts_prev = detected_labels_src_pts.copy()  # Save current detected keypoint coordinates for next frame

        return h, detected_labels_prev, detected_labels_src_pts_prev

    def extract_boxes_p(self, tracks_players, frame_nbr):
        list_boxes = []  # Initialize an empty list to store player bounding boxes
        colors=[]
        player_dict = tracks_players[frame_nbr]  # Assuming self.tracks_players is structured correctly

        for track_id, player in player_dict.items():
            list_boxes.append(np.array(player["bbox"])) 
            colors.append(np.array(player['team_color'])) # Convert bbox to numpy array and append to list

        return list_boxes,colors
    def extract_boxes_b(self, tracks, frame_nbr):
      if frame_nbr in tracks['ball'] and 1 in tracks['ball'][frame_nbr]:
        return tracks['ball'][frame_nbr][1]['bbox']
      else:
        # Gérer le cas où la clé n'existe pas
        return None


    def draw_on_tactic_map(self, tac_map, player_positions, ball_position, ball_track_history, colors):
    # Drawing logic (you can update this as needed)
      for pos, color in zip(player_positions, colors):
        cv2.circle(tac_map, tuple(int(v) for v in pos), 5, color, -1)
      if ball_position is not None:
        cv2.circle(tac_map, tuple(int(v) for v in ball_position[:2]), 5, (0, 255, 0), -1)
      for pos in ball_track_history['dst']:
        cv2.circle(tac_map, pos, 5, (0, 0, 255), -1)
      return tac_map

    def classes_nammes(self,yaml_path):
        # Get football field keypoints numerical to alphabetical mapping
      
        with open(yaml_path, 'r') as file:
          classes_names_dic = yaml.safe_load(file)
          classes_names_dic = classes_names_dic['names']
        return classes_names_dic
    def ajout_adjusted_position(self, pred_dst_pts, frame_nbr):
       
        for i, (track_id, track_info) in enumerate(self.tracks["players"][frame_nbr].items()):
            adjusted_position = pred_dst_pts[i]
            self.tracks["players"][frame_nbr][track_id]["adjusted_position_chat"] = adjusted_position.tolist()

    def ajout_adjusted_position_ball(self,detected_ball_dst_pos,frame_nbr):
        self.tracks["ball"][frame_nbr][1]["adjusted_position_chat"] = detected_ball_dst_pos.tolist()
           

    def giv_tracks(self):
        return self.tracks
        

        
