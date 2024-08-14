import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import get_center_of_bbox,measure_distance
sys.path.append('../')

class FootballArmDetection:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)
        self.mp_drawing = mp.solutions.drawing_utils

    def get_arm_player(self, frame, player_bbox):
      
        image_height, image_width, _ = frame.shape
        image = frame[int(player_bbox[1])-20:int(player_bbox[3]+20), int(player_bbox[0])-20:int(player_bbox[2]+20)]
        # Convertir l'image en RGB (Mediapipe nécessite des images RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Détecter la pose complète dans l'image
        with self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            # Convertir l'image RGB pour le traitement par Mediapipe
            results = pose.process(image_rgb)

            # Vérifier si une pose complète est détectée
            if not results.pose_landmarks:
                
                return None,image_height, image_width
            
            # Identifier et retourner les landmarks pour la partie supérieure du corps
            upper_body_landmarks = {
                "right_shoulder": results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                "left_shoulder": results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                "right_elbow": results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
                "left_elbow": results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                "right_wrist": results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST],
                "left_wrist": results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST],
            }
            # Dessiner les landmarks sur l'image originale
            # Créer le nom de fichier avec le chemin complet vers le dossier "image"
            filename = f"image/landmarks_frame_{int(player_bbox[1])}.jpg"

# Enregistrer l'image avec les landmarks
            cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            

        return upper_body_landmarks,image_height, image_width
    
    def is_handball(self, arm_landmarks, ball_bbox, frame):
        # Vérifier si la balle est en contact avec les bras
        if not ball_bbox:
            return False, frame
        
        width = frame.shape[1]
        height = frame.shape[0]
        
        center = get_center_of_bbox(ball_bbox)
        ball_x = center[0]
        ball_y = center[1]
        image_height, image_width, _ = frame.shape
        
        contact_threshold = 10 # Ajuster le seuil de contact selon les besoins

        for key, landmark in arm_landmarks.items():
            x = int(landmark.x * image_width)  # Adapter aux dimensions de l'image complète
            y = int(landmark.y * image_height)  # Adapter aux dimensions de l'image complète
            if abs(x - ball_x) < contact_threshold and abs(y - ball_y) < contact_threshold:
                # Dessiner un cercle autour de la balle et des points de contact
                cv2.circle(frame, (ball_x, ball_y), 5, (0, 255, 0), -1)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                return True, frame
        
        return False, frame
    
    
    def plot_ball_trajectory_and_landmarks(self, track_ball, right_shoulder, left_shoulder, right_elbow, left_elbow, right_wrist, left_wrist):
        ball_positions = []
        for frame_data in track_ball:
          if frame_data:
            ball_data = frame_data.get(1)
            if ball_data:
                ball_bbox = ball_data['bbox']
                # Calculer le centre de la bbox de la balle (utiliser les coordonnées x et y)
                x_center = (ball_bbox[0] + ball_bbox[2]) / 2  # (x_min + x_max) / 2
                y_center = (ball_bbox[1] + ball_bbox[3]) / 2  # (y_min + y_max) / 2
                ball_positions.append((x_center, y_center))

    # Extraire les coordonnées x et y des positions de la balle
        x_positions = [pos[0] for pos in ball_positions]
        y_positions = [pos[1] for pos in ball_positions]

    # Créer le graphe
        plt.figure(figsize=(10, 8))
        
    # Tracer la trajectoire de la balle
        plt.plot(x_positions, y_positions, marker='o', linestyle='-', color='b', label='Trajectoire de la balle')

    # Tracer les trajectoires des landmarks des bras des joueurs
        if right_shoulder:
         plt.plot([pos[0] for pos in right_shoulder], [pos[1] for pos in right_shoulder], marker='o', linestyle='-', color='r', label='Right Shoulder')
        if left_shoulder:
           plt.plot([pos[0] for pos in left_shoulder], [pos[1] for pos in left_shoulder], marker='o', linestyle='-', color='g', label='Left Shoulder')
        if right_elbow:
          plt.plot([pos[0] for pos in right_elbow], [pos[1] for pos in right_elbow], marker='o', linestyle='-', color='c', label='Right Elbow')
        if left_elbow:
             plt.plot([pos[0] for pos in left_elbow], [pos[1] for pos in left_elbow], marker='o', linestyle='-', color='m', label='Left Elbow')
        if right_wrist:
           plt.plot([pos[0] for pos in right_wrist], [pos[1] for pos in right_wrist], marker='o', linestyle='-', color='y', label='Right Wrist')
        if left_wrist:
           plt.plot([pos[0] for pos in left_wrist], [pos[1] for pos in left_wrist], marker='o', linestyle='-', color='k', label='Left Wrist')

        plt.title('Trajectoire de la balle et des landmarks des bras')
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    def plot_ball_trajectory_and_landmarks_second(self, ax, track_ball, right_shoulder, left_shoulder, right_elbow, left_elbow, right_wrist, left_wrist):
     ball_positions = []
     for frame_data in track_ball:
        if frame_data:
            ball_data = frame_data.get(1)
            if ball_data:
                ball_bbox = ball_data['bbox']
                x_center = (ball_bbox[0] + ball_bbox[2]) / 2
                y_center = (ball_bbox[1] + ball_bbox[3]) / 2
                ball_positions.append((x_center, y_center))

     if ball_positions:
        ball_positions = np.array(ball_positions)
        ax.plot(ball_positions[:, 0], ball_positions[:, 1], 'bo-', label='Ball Trajectory')
     if right_shoulder:
         plt.plot([pos[0] for pos in right_shoulder], [pos[1] for pos in right_shoulder], marker='o', linestyle='-', color='r', label='Right Shoulder')
     if left_shoulder:
           plt.plot([pos[0] for pos in left_shoulder], [pos[1] for pos in left_shoulder], marker='o', linestyle='-', color='g', label='Left Shoulder')
     if right_elbow:
          plt.plot([pos[0] for pos in right_elbow], [pos[1] for pos in right_elbow], marker='o', linestyle='-', color='c', label='Right Elbow')
     if left_elbow:
             plt.plot([pos[0] for pos in left_elbow], [pos[1] for pos in left_elbow], marker='o', linestyle='-', color='m', label='Left Elbow')
     if right_wrist:
           plt.plot([pos[0] for pos in right_wrist], [pos[1] for pos in right_wrist], marker='o', linestyle='-', color='y', label='Right Wrist')
     if left_wrist:
           plt.plot([pos[0] for pos in left_wrist], [pos[1] for pos in left_wrist], marker='o', linestyle='-', color='k', label='Left Wrist')

     ax.legend()
     ax.set_xlabel('X Position')
     ax.set_ylabel('Y Position')
     ax.set_title('Ball Trajectory and Player Arm Landmarks')

      
   
