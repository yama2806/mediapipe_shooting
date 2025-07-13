import cv2
import numpy as np
import mediapipe as mp
import pygame
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class Bullet:
    "弾丸を管理するクラス"
    def __init__():
        
class Enemy:
    "敵を管理するクラス"

class Player:
    "敵を管理するクラス"

class HandTracker:
    """
    MediaPipeによる手の検出と、指の座標+ジェスチャー分類を行うクラス
    """

    def __init__(self, model_path='hand_gesture_dataset.csv'):
        # MediaPipeの初期化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # ランダムフォレストモデルの読み込みと学習
        self.model = self._load_model(model_path)

        # 情報保持用の変数（他クラスから参照可能）
        self.last_landmarks = None        # 指の座標（21点の(x, y, z)）
        self.last_prediction = None       # 予測されたラベル（int）
        self.last_confidence = 0.0        # 予測の信頼度（float）

    def _load_model(self, path):
        """CSVから学習済みのランダムフォレストモデルを読み込む"""
        df = pd.read_csv(path)
        X = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']]
        y = df['target']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    def process_frame(self, frame):
        """
        カメラフレームを処理して手の検出とジェスチャー分類を行う。
        返り値:
            - pred（int or None）: 予測されたジェスチャーラベル（信頼度が低い場合はNone）
            - confidence（float）: 信頼度（0〜1）
        """
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        self.last_landmarks = None
        self.last_prediction = None
        self.last_confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 各指の座標を保存（x, y, z）形式
                self.last_landmarks = [
                    (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
                ]

                # 特徴量を抽出して予測
                features = self._extract_features(hand_landmarks)
                if features is not None:
                    probs = self.model.predict_proba([features])[0]
                    pred = np.argmax(probs)
                    confidence = probs[pred]

                    if confidence >= 0.75:
                        self.last_prediction = pred
                        self.last_confidence = confidence
                        return pred, confidence  # 十分信頼できる予測
                    else:
                        self.last_prediction = None
                        self.last_confidence = confidence
                        return None, confidence  # 信頼度低い
        return None, 0.0  # 手が検出されなかった

    def _extract_features(self, hand_landmarks):
        """
        手のランドマークから7つの特徴量を抽出する（分類器の入力用）
        """
        landmarks = hand_landmarks.landmark
        base = np.array([landmarks[0].x, landmarks[0].y])
        features = []

        # 各指先（4, 8, 12, 16, 20）と手首（0）の距離
        for i in [4, 8, 12, 16, 20]:
            tip = np.array([landmarks[i].x, landmarks[i].y])
            dist = np.linalg.norm(tip - base)
            features.append(dist)

        # 親指の角度（x軸との余弦）
        thumb_base = np.array([landmarks[1].x, landmarks[1].y])
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        thumb_vec = thumb_tip - thumb_base
        if np.linalg.norm(thumb_vec) != 0:
            thumb_vec /= np.linalg.norm(thumb_vec)
        else:
            thumb_vec = np.array([0, 0])
        cos_theta = np.dot(thumb_vec, np.array([1, 0]))
        features.append(cos_theta)

        # 親指のz方向の差
        thumb_z_diff = landmarks[4].z - landmarks[1].z
        features.append(thumb_z_diff)

        if len(features) == 7:
            return features
        return None



class Game:
    "ゲーム全体の進行を管理するクラス"
    def __init__(self):
        "Pygameの初期化"

    def run(self):
        "メインループ"

    def update(self):
        "全てのゲームオブジェクトの状態を更新する"
    
    def handle_player_turn(self):
        "プレイヤーのターンの処理"

    def handle_enemy_turn(self):
        "敵のターンの処理"