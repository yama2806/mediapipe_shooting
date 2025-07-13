import pygame
import random
import math
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# --- 定数 ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)



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
                    input_df = pd.DataFrame([features], columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
                    probs = self.model.predict_proba(input_df)[0]
                    pred = np.argmax(probs)
                    confidence = probs[pred]

                    if confidence >= 0.6:
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



class Bullet(pygame.sprite.Sprite):
    """プレイヤーの弾を管理するクラス"""
    def __init__(self, start_pos, target_pos):
        super().__init__()
        self.image = pygame.Surface((10, 10))
        self.image.fill(YELLOW)
        self.float_x = float(start_pos [0])
        self.float_y = float(start_pos [1])
        self.rect = self.image.get_rect(center=(self.float_x, self.float_y))
        angle = math.atan2(target_pos [1] - start_pos [1], target_pos [0] - start_pos [0])
        speed = 15
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.float_x += self.vx
        self.float_y += self.vy
        self.rect.centerx = round(self.float_x)
        self.rect.centery = round(self.float_y)
        if not pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT).colliderect(self.rect):
            self.kill()

class Enemy(pygame.sprite.Sprite):
    """敵を管理するクラス"""
    def __init__(self, pos): # 引数にposを追加
        super().__init__()
        try:
            self.image = pygame.image.load("images/enemy.png").convert_alpha()
            self.image = pygame.transform.scale(self.image, (100, 100)) # サイズを100x100に変更
        except pygame.error:
            self.image = pygame.Surface((50, 50)) # サイズを50x50に変更
            self.image.fill(RED)

        self.rect = self.image.get_rect(center=pos) # 受け取った位置で初期化
        self.max_health = 100
        self.health = self.max_health
        self.speed_x = random.choice([-3, -2, 2, 3]) # スピードもランダムに

    def update(self):
        """敵の動きを更新する"""
        self.rect.x += self.speed_x
        if self.rect.left < 0 or self.rect.right > SCREEN_WIDTH:
            self.speed_x *= -1

    def draw(self, screen):
        """敵の画像とHPバーを描画する"""
        screen.blit(self.image, self.rect)
        health_bar_width = int(self.rect.width * (self.health / self.max_health))
        hp_bar_rect = pygame.Rect(self.rect.left, self.rect.top - 15, self.rect.width, 8)
        hp_bar_fill_rect = pygame.Rect(self.rect.left, self.rect.top - 15, health_bar_width, 8)
        pygame.draw.rect(screen, RED, hp_bar_rect)
        pygame.draw.rect(screen, GREEN, hp_bar_fill_rect)

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.kill() # HPが0になったら自身をグループから削除
            return True
        return False


class Player(pygame.sprite.Sprite):
    """プレイヤーを管理するクラス"""
    def __init__(self):
        super().__init__()
        try:
            self.image = pygame.image.load("images/player.png").convert_alpha()
            self.image = pygame.transform.scale(self.image, (180, 180)) # 表示サイズ
        except pygame.error:
            self.image = pygame.Surface((100, 150))
            self.image.fill(BLUE)

        self.rect = self.image.get_rect(midbottom=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 10))
        self.max_health = 100
        self.health = self.max_health

    def draw(self, screen):
        """プレイヤーを画面に描画する"""
        screen.blit(self.image, self.rect)

    def take_damage(self, amount):
        self.health -= amount
        if self.health < 0:
            self.health = 0
            return True
        return False

class Barrier(pygame.sprite.Sprite):
    """手のひらで操作する円形のバリア"""
    def __init__(self, handtracker):
        super().__init__()
        self.handtracker = handtracker
        self.radius = 60  # バリアの半径
        # 画像ではなく、円を描画するための設定
        self.image = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (0, 100, 255, 150), (self.radius, self.radius), self.radius)
        self.rect = self.image.get_rect()

    def update(self):
        """バリアの位置をての位置に合わせる"""
        if self.handtracker.last_prediction == 2:
            palm = self.handtracker.last_landmarks[0]
            middle_root = self.handtracker.last_landmarks[9]

            center_x = (palm[0] + middle_root[0]) / 2 * SCREEN_WIDTH
            center_y = (palm[1] + middle_root[1]) / 2 * SCREEN_HEIGHT

            self.rect.center = (int(center_x), int(center_y))

        else:
            self.rect.center = (-9999, -9999)
    

    def draw(self, screen):
        """バリアを描画する"""
        screen.blit(self.image, self.rect)

class Game:
    """ゲーム全体の進行を管理するクラス"""
    def __init__(self, handtracker):
        pygame.init()
        pygame.mixer.init()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Multi-Enemy Shooting Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 50)
        self.running = True
        self.handtracker = handtracker
        self.prev_hand = None

        self.reset_game()

        try:
            self.shoot_sound = pygame.mixer.Sound("sounds/shoot.wav")
            self.hit_sound = pygame.mixer.Sound("sounds/hit.wav")
            self.guard_sound = pygame.mixer.Sound("sounds/guard.wav")
            self.shoot_sound.set_volume(0.5)
            self.hit_sound.set_volume(0.5)
            self.guard_sound.set_volume(0.5)
        except pygame.error as e:
            print(f"効果音ファイルが見つかりません: {e}")
            self.shoot_sound = self.hit_sound = self.guard_sound = None

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        if self.cap.isOpened():
            self.cap.release()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if self.game_state == "START":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.game_state = "PLAYING"
                    self.turn = "PLAYER_TURN"
                    self.turn_timer = 0
                    self.spawn_enemies(3)
                    # ▼▼▼ 変更点: BGMファイル名を変更 ▼▼▼
                    try:
                        pygame.mixer.music.load("sounds/battle.mp3")
                        pygame.mixer.music.play(-1)
                    except pygame.error as e:
                        print(f"BGMファイルが見つかりません: {e}")
                    # ▲▲▲ ここまで変更 ▲▲▲

            elif self.game_state == "PLAYING":
                if self.turn == "PLAYER_TURN":
                    ret, frame = self.cap.read()
                    if ret:
                        pred, conf = self.handtracker.process_frame(frame)
                        print(f"予測ラベル: {pred}, 信頼度: {conf}")

                    if pred == 1 and self.prev_hand != 1:
                        if self.handtracker.last_landmarks:
                            pos = self.handtracker.last_landmarks[8]
                            screen_x = int(pos[0] * SCREEN_WIDTH)
                            screen_y = int(pos[1] * SCREEN_HEIGHT)
                            self.handle_player_attack((screen_x, screen_y))
            
                    self.prev_hand = pred
                    
            
            elif self.game_state in ("CLEAR", "OVER"):
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.reset_game()

    def reset_game(self):
        """ゲームの状態をリセットする"""
        pygame.mixer.music.stop()
        self.player = Player()
        self.enemies = pygame.sprite.Group()
        self.bullets = pygame.sprite.Group()
        self.game_state = "START"
        self.turn = ""
        self.message = ""
        self.message_timer = 0
        self.turn_timer = 0
        
        self.handtracker = HandTracker()
        self.barrier = Barrier(self.handtracker)
        self.enemy_attacks = []
        self.enemy_attack_count = 0
        self.attack_spawn_timer = 0
        self.cap = cv2.VideoCapture(0)

    def spawn_enemies(self, count):
        self.enemies.empty()
        spawn_attempts = 0
        for _ in range(count):
            while True:
                x = random.randint(50, SCREEN_WIDTH - 50)
                y = random.randint(50, SCREEN_HEIGHT // 2)
                new_enemy = Enemy((x, y))
                
                collided_enemies = pygame.sprite.spritecollide(new_enemy, self.enemies, False)
                if not collided_enemies:
                    self.enemies.add(new_enemy)
                    break
                
                spawn_attempts += 1
                if spawn_attempts > 100:
                    print("敵の配置に失敗しました。")
                    return

    def update(self):
        if self.game_state == "PLAYING":
            self.turn_timer += self.clock.get_time() / 1000

            if self.turn == "PLAYER_TURN":
                self.enemies.update()
                self.bullets.update()
                
                for bullet in self.bullets:
                    hit_enemies = pygame.sprite.spritecollide(bullet, self.enemies, False)
                    if hit_enemies:
                        if self.hit_sound: self.hit_sound.play()
                        if hit_enemies[0].take_damage(20):
                            if not self.enemies:
                                self.game_state = "CLEAR"
                                self.set_message("GAME CLEAR!")
                                # ▼▼▼ 変更点: BGMファイル名を変更 ▼▼▼
                                pygame.mixer.music.stop()
                                try:
                                    pygame.mixer.music.load("sounds/clear.mp3")
                                    pygame.mixer.music.play(-1)
                                except pygame.error as e:
                                    print(f"BGMファイルが見つかりません: {e}")
                                # ▲▲▲ ここまで変更 ▲▲▲
                        bullet.kill()

                if self.turn_timer > 10 and self.game_state == "PLAYING":
                    self.turn = "ENEMY_TURN"
                    self.set_message("ENEMY TURN")
                    self.turn_timer = 0
                    self.enemy_attack_count = 0
                    self.attack_spawn_timer = 1500

            elif self.turn == "ENEMY_TURN":
                ret, frame = self.cap.read()
                if ret:
                    self.handtracker.process_frame(frame)
                self.barrier.update()


                if self.enemies and self.enemy_attack_count < 5 * len(self.enemies):
                    self.attack_spawn_timer += self.clock.get_time()
                    if self.attack_spawn_timer > 1000:
                        self.attack_spawn_timer = 0
                        pos = (random.randint(100, SCREEN_WIDTH - 100), random.randint(100, SCREEN_HEIGHT - 100))
                        self.enemy_attacks.append({"pos": pos, "timer": 0, "duration": 2000, "max_radius": 100})
                        self.enemy_attack_count += 1
                
                for attack in self.enemy_attacks[:]:
                    attack["timer"] += self.clock.get_time()
                    if attack["timer"] >= attack["duration"]:
                        distance = math.hypot(attack["pos"][0] - self.barrier.rect.centerx, attack["pos"][1] - self.barrier.rect.centery)
                        if distance < self.barrier.radius:
                            self.set_message("GUARD SUCCESS!")
                            if self.guard_sound: self.guard_sound.play()
                        else:
                            self.set_message("DAMAGE!")
                            if self.hit_sound: self.hit_sound.play()
                            if self.player.take_damage(10):
                                self.game_state = "OVER"
                                self.set_message("GAME OVER!")
                                # ▼▼▼ 変更点: BGMファイル名を変更 ▼▼▼
                                pygame.mixer.music.stop()
                                try:
                                    pygame.mixer.music.load("sounds/lose.mp3")
                                    pygame.mixer.music.play(-1)
                                except pygame.error as e:
                                    print(f"BGMファイルが見つかりません: {e}")
                                # ▲▲▲ ここまで変更 ▲▲▲
                        self.enemy_attacks.remove(attack)
                
                attack_limit = 5 * len(self.enemies) if self.enemies else 0
                if self.enemy_attack_count >= attack_limit and not self.enemy_attacks:
                    if not self.enemies:
                        self.game_state = "CLEAR"
                        self.set_message("GAME CLEAR!")
                    else:
                        self.turn = "PLAYER_TURN"
                        self.set_message("PLAYER TURN")
                        self.turn_timer = 0
                        self.spawn_enemies(3)

    def handle_player_attack(self, target_pos):
        player_gun_pos = (self.player.rect.centerx, self.player.rect.centery - 20)
        new_bullet = Bullet(player_gun_pos, target_pos)
        self.bullets.add(new_bullet)
        if self.shoot_sound: self.shoot_sound.play()

    # Gameクラスのdrawメソッド
    def draw(self):
        try:
            background = pygame.image.load("images/stage.png").convert()
            background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
            self.screen.blit(background, (0, 0))
        except pygame.error as e:
            print(f"背景画像 (images/stage.png) が見つかりません: {e}")
            self.screen.fill(BLACK) # エラー時は黒で塗りつぶす

        if self.game_state == "START":
            title_text = self.font.render("Multi-Enemy Shooting Game", True, WHITE)
            start_text = self.font.render("PRESS SPACE TO START", True, WHITE)
            self.screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, SCREEN_HEIGHT // 3))
            self.screen.blit(start_text, (SCREEN_WIDTH // 2 - start_text.get_width() // 2, SCREEN_HEIGHT // 2))

        elif self.game_state == "PLAYING":
            for enemy in self.enemies:
                enemy.draw(self.screen)

            self.player.draw(self.screen)
            player_hp_text = self.font.render(f"HP: {self.player.health}", True, WHITE)
            self.screen.blit(player_hp_text, (20, SCREEN_HEIGHT - 50))
            self.bullets.draw(self.screen)

            turn_text = self.font.render(f"TURN: {self.turn}", True, YELLOW)
            self.screen.blit(turn_text, (SCREEN_WIDTH // 2 - turn_text.get_width() // 2, 10))

            if self.turn == 'PLAYER_TURN':
                time_left = max(0, 10 - self.turn_timer)
                timer_text = self.font.render(f"TIME: {time_left:.1f}", True, WHITE)
                self.screen.blit(timer_text, (SCREEN_WIDTH - 200, 10))

                if self.turn == "PLAYER_TURN" and self.handtracker.last_prediction == 0:
                    if self.handtracker.last_landmarks:
                        fingertip = self.handtracker.last_landmarks[8]
                        fx = int(fingertip[0] * SCREEN_WIDTH)
                        fy = int(fingertip[1] * SCREEN_HEIGHT)
                        pygame.draw.circle(self.screen, YELLOW, (fx, fy), 10)  # 半径10の黄色い丸


            if self.turn == "ENEMY_TURN":
                self.barrier.draw(self.screen)
                for attack in self.enemy_attacks:
                    progress = min(attack["timer"] / attack["duration"], 1.0)
                    current_radius = int(attack["max_radius"] * (1 - progress))
                    if current_radius > 0:
                        pygame.draw.circle(self.screen, RED, attack["pos"], current_radius, 5)

            elif self.turn == "PLAYER_TURN":
                pygame.mouse.set_visible(True)

        if self.message and pygame.time.get_ticks() - self.message_timer < 2000:
            msg_surf = self.font.render(self.message, True, WHITE)
            self.screen.blit(msg_surf, (SCREEN_WIDTH // 2 - msg_surf.get_width() // 2, SCREEN_HEIGHT // 2))

        elif self.game_state == "CLEAR":
            clear_text = self.font.render("GAME CLEAR!", True, GREEN)
            restart_text = self.font.render("PRESS SPACE TO RESTART", True, WHITE)
            self.screen.blit(clear_text, (SCREEN_WIDTH // 2 - clear_text.get_width() // 2, SCREEN_HEIGHT // 3))
            self.screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2))

        elif self.game_state == "OVER":
            over_text = self.font.render("GAME OVER!", True, RED)
            restart_text = self.font.render("PRESS SPACE TO RESTART", True, WHITE)
            self.screen.blit(over_text, (SCREEN_WIDTH // 2 - over_text.get_width() // 2, SCREEN_HEIGHT // 3))
            self.screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2))
            
    def set_message(self, text):
        self.message = text
        self.message_timer = pygame.time.get_ticks()

if __name__ == "__main__":
    tracker = HandTracker()
    game = Game(tracker)
    game.run()
    