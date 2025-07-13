import pygame
import random
import math

# --- 定数 ---
# 画面サイズ
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
# 色の定義
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 200, 200)

# --- クラス定義 ---

class BeamEffect:
    """攻撃ビームのエフェクトを管理するクラス"""
    def __init__(self, start_pos, end_pos, color=CYAN, is_blocked=False):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.color = color
        self.is_blocked = is_blocked  # ガードされたかどうかのフラグ
        self.lifetime = 0.25 if is_blocked else 0.15 # ガード時は少し長く表示
        self.timer = 0.0

    def update(self, delta_time):
        """エフェクトの生存時間を更新し、時間切れならFalseを返す"""
        self.timer += delta_time
        return self.timer < self.lifetime

    def draw(self, screen):
        """ビームを描画する"""
        progress = self.timer / self.lifetime
        alpha = max(0, 255 * (1 - progress))
        
        # ガード成功時はビームが途中で止まり、火花が散る
        if self.is_blocked:
            draw_end_pos = self.end_pos
            # sin波を使って明滅する火花を表現
            spark_radius = int(25 * math.sin(progress * math.pi)) 
            if spark_radius > 0:
                pygame.draw.circle(screen, YELLOW, self.end_pos, spark_radius)
                pygame.draw.circle(screen, WHITE, self.end_pos, spark_radius // 2)
        else:
            draw_end_pos = self.end_pos

        # ビーム本体の描画
        try:
            # NOTE: 一部のPygame環境ではアルファ付きのline描画が直接できない場合がある
            pygame.draw.line(screen, (*self.color, int(alpha)), self.start_pos, draw_end_pos, 8)
        except (TypeError, ValueError):
             pygame.draw.line(screen, self.color, self.start_pos, draw_end_pos, 8)
        pygame.draw.line(screen, WHITE, self.start_pos, draw_end_pos, 2)

class FloatingScore(pygame.sprite.Sprite):
    """ヒット時にスコアを浮かび上がらせるクラス"""
    def __init__(self, pos, score):
        super().__init__()
        self.font = pygame.font.Font(None, 40)
        self.image = self.font.render(f"+{score}", True, YELLOW)
        self.rect = self.image.get_rect(center=pos)
        self.timer = 0.0
        self.lifetime = 0.8

    def update(self, delta_time):
        """スコアを上昇させながらフェードアウトさせる"""
        self.timer += delta_time
        self.rect.y -= 1
        alpha = max(0, 255 * (1 - (self.timer / self.lifetime)))
        self.image.set_alpha(int(alpha))
        if self.timer >= self.lifetime:
            self.kill()

class Enemy(pygame.sprite.Sprite):
    """敵を管理するクラス"""
    def __init__(self, pos):
        super().__init__()
        try:
            self.base_image = pygame.transform.scale(pygame.image.load("images/enemy.png").convert_alpha(), (100, 100))
        except pygame.error:
            self.base_image = pygame.Surface((100, 100), pygame.SRCALPHA)
            pygame.draw.circle(self.base_image, RED, (50, 50), 50)
        
        self.image = self.base_image.copy()
        self.rect = self.image.get_rect(center=pos)
        self.max_health = 100
        self.health = self.max_health
        
        # 動きの初期設定
        self.direction = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
        self.speed = random.uniform(2.0, 4.0)
        self.direction_change_timer = 0.0
        self.direction_change_interval = random.uniform(1.5, 3.0)
        
        # 拡大・縮小の初期設定
        self.scale = 1.0
        self.scale_direction = 1
        self.scale_speed = 0.01

    def update(self, delta_time):
        """敵の移動と拡大・縮小を更新する"""
        # --- 移動処理 ---
        self.rect.move_ip(self.direction * self.speed)

        # 壁での反射
        if self.rect.left < 0 or self.rect.right > SCREEN_WIDTH:
            self.direction.x *= -1
        if self.rect.top < 0 or self.rect.bottom > SCREEN_HEIGHT // 2:
            self.direction.y *= -1
        
        # --- ランダムな方向転換 ---
        self.direction_change_timer += delta_time
        if self.direction_change_timer > self.direction_change_interval:
            self.direction_change_timer = 0
            self.direction_change_interval = random.uniform(1.5, 3.0)
            self.direction = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
            self.speed = random.uniform(2.0, 5.0)

        # --- 拡大・縮小処理 ---
        self.scale += self.scale_direction * self.scale_speed
        if self.scale > 1.3 or self.scale < 0.7:
            self.scale_direction *= -1
        
        center_pos = self.rect.center
        new_size = (int(self.base_image.get_width() * self.scale), int(self.base_image.get_height() * self.scale))
        self.image = pygame.transform.scale(self.base_image, new_size)
        self.rect = self.image.get_rect(center=center_pos)

    def draw(self, screen):
        """敵本体とHPバーを描画する"""
        screen.blit(self.image, self.rect)
        health_bar_width = int(self.rect.width * (self.health / self.max_health))
        hp_bar_background_rect = pygame.Rect(self.rect.left, self.rect.top - 15, self.rect.width, 8)
        hp_bar_fill_rect = pygame.Rect(self.rect.left, self.rect.top - 15, health_bar_width, 8)
        pygame.draw.rect(screen, RED, hp_bar_background_rect)
        pygame.draw.rect(screen, GREEN, hp_bar_fill_rect)

    def take_damage(self, amount):
        """ダメージを受けてHPを減らす"""
        self.health -= amount
        if self.health <= 0:
            self.kill()
            return True
        return False

class Player(pygame.sprite.Sprite):
    """プレイヤー（画面下のキャラクター）を管理するクラス"""
    def __init__(self):
        super().__init__()
        try:
            self.image = pygame.transform.scale(pygame.image.load("images/player.png").convert_alpha(), (180, 180))
        except pygame.error:
            self.image = pygame.Surface((100, 150))
            self.image.fill(BLUE)
        self.rect = self.image.get_rect(midbottom=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 10))
        self.max_health = 100
        self.health = self.max_health
        
    def draw(self, screen):
        screen.blit(self.image, self.rect)
        
    def take_damage(self, amount):
        self.health -= amount
        if self.health < 0:
            self.health = 0
        return self.health == 0

class Barrier(pygame.sprite.Sprite):
    """マウスで操作する円形のバリア"""
    def __init__(self):
        super().__init__()
        self.radius = 60
        self.image = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (0, 100, 255, 150), (self.radius, self.radius), self.radius)
        self.rect = self.image.get_rect()
        
    def update(self):
        self.rect.center = pygame.mouse.get_pos()
        
    def draw(self, screen):
        screen.blit(self.image, self.rect)

def calculate_score(hit_pos, enemy):
    """命中精度と敵のサイズからスコアを計算する関数"""
    distance = math.hypot(hit_pos[0] - enemy.rect.centerx, hit_pos[1] - enemy.rect.centery)
    max_distance = enemy.rect.width / 2
    accuracy_score = max(0, 100 * (1 - distance / max_distance))
    size_score = 50 * (1 / enemy.scale)
    return int(accuracy_score + size_score)

class Game:
    """ゲーム全体の進行を管理するクラス"""
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("最終改善版 シューティング")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 50)
        self.running = True
        pygame.mouse.set_visible(True)

        self.load_assets()
        self.reset_game()

    def load_assets(self):
        """画像や音声ファイルを一度に読み込む"""
        try:
            self.background_image = pygame.transform.scale(pygame.image.load("images/stage.png").convert(), (SCREEN_WIDTH, SCREEN_HEIGHT))
        except pygame.error:
            self.background_image = None
        try:
            self.shoot_sound = pygame.mixer.Sound("sounds/shoot.wav")
            self.hit_sound = pygame.mixer.Sound("sounds/hit.wav")
            self.guard_sound = pygame.mixer.Sound("sounds/guard.wav")
            self.shoot_sound.set_volume(0.5)
            self.hit_sound.set_volume(0.5)
            self.guard_sound.set_volume(0.5)
        except pygame.error:
            self.shoot_sound = self.hit_sound = self.guard_sound = None

    def play_bgm(self, track_name):
        """BGMを再生する"""
        pygame.mixer.music.stop()
        try:
            pygame.mixer.music.load(f"sounds/{track_name}.mp3")
            pygame.mixer.music.play(-1)
        except pygame.error:
            pass

    def run(self):
        """メインループ"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
        pygame.quit()
        
    def handle_events(self):
        """キーボードやマウスのイベントを処理する"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if self.game_state == "START":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.game_state = "PLAYING"
                    self.set_turn("PLAYER_TURN")
                    self.spawn_enemies(3)
                    self.play_bgm("battle")
            
            elif self.game_state == "PLAYING" and self.turn == "PLAYER_TURN":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_player_attack(event.pos)
            
            elif self.game_state in ("CLEAR", "OVER"):
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.reset_game()

    def reset_game(self):
        """ゲームの状態を初期化する"""
        self.player = Player()
        self.enemies = pygame.sprite.Group()
        self.effects = []
        self.floating_scores = pygame.sprite.Group()
        self.enemy_attacks = []
        
        self.game_state = "START"
        self.turn = ""
        self.message = ""
        self.score = 0
        self.message_timer = 0
        self.turn_timer = 0
        self.enemy_attack_count = 0
        self.attack_spawn_timer = 0
        
        self.barrier = Barrier()
        
    def spawn_enemies(self, count):
        """指定された数の敵を生成する"""
        self.enemies.empty()
        for _ in range(count):
            while True:
                x = random.randint(50, SCREEN_WIDTH - 50)
                y = random.randint(50, SCREEN_HEIGHT // 2)
                new_enemy = Enemy((x, y))
                if not pygame.sprite.spritecollide(new_enemy, self.enemies, False):
                    self.enemies.add(new_enemy)
                    break

    def update(self):
        """ゲームの状態をフレームごとに更新する"""
        delta_time = self.clock.tick(60) / 1000.0
        if self.game_state != "PLAYING":
            return

        self.turn_timer += delta_time
        
        for effect in self.effects[:]:
            if not effect.update(delta_time):
                self.effects.remove(effect)
        self.floating_scores.update(delta_time)

        if self.turn == "PLAYER_TURN":
            self.enemies.update(delta_time)
            if self.turn_timer > 10:
                self.set_turn("ENEMY_TURN")
        
        elif self.turn == "ENEMY_TURN":
            self.barrier.update()
            
            # 敵の攻撃を生成
            max_attacks = 5 * len(self.enemies)
            if self.enemies and self.enemy_attack_count < max_attacks:
                self.attack_spawn_timer += delta_time * 1000
                if self.attack_spawn_timer > 1000:
                    self.attack_spawn_timer = 0
                    # 攻撃予測円はランダムな位置に出現（ゲームロジック）
                    attack_pos = (random.randint(100, SCREEN_WIDTH - 100), random.randint(100, SCREEN_HEIGHT - 100))
                    self.enemy_attacks.append({"pos": attack_pos, "timer": 0, "duration": 2000, "max_radius": 100})
                    self.enemy_attack_count += 1
            
            # 敵の攻撃の着弾処理
            for attack in self.enemy_attacks[:]:
                attack["timer"] += delta_time * 1000
                if attack["timer"] >= attack["duration"]:
                    attacker = None
                    if self.enemies:
                        attacker = random.choice(self.enemies.sprites())
                    
                    # 防御判定は予測円の中心(attack["pos"])とバリアで行う
                    is_guarded = math.hypot(attack["pos"][0] - self.barrier.rect.centerx, attack["pos"][1] - self.barrier.rect.centery) < self.barrier.radius

                    if is_guarded:
                        self.set_message("GUARD SUCCESS!")
                        if self.guard_sound: self.guard_sound.play()
                        
                        if attacker:
                            # 敵からバリアまでのベクトル
                            vec_to_shield = pygame.Vector2(self.barrier.rect.center) - pygame.Vector2(attacker.rect.center)
                            # バリアの円周上でビームが止まる位置
                            impact_pos = pygame.Vector2(attacker.rect.center) + vec_to_shield.normalize() * (vec_to_shield.length() - self.barrier.radius)
                            # ガード成功エフェクト
                            self.effects.append(BeamEffect(attacker.rect.center, impact_pos, color=YELLOW, is_blocked=True))
                    else:
                        self.set_message("DAMAGE!")
                        if self.hit_sound: self.hit_sound.play()
                        
                        # ダメージ時のビームは敵からプレイヤーに向かって飛ぶように見せる
                        if attacker:
                            visual_target_pos = self.player.rect.center
                            self.effects.append(BeamEffect(attacker.rect.center, visual_target_pos, color=RED))
                        
                        if self.player.take_damage(10):
                            self.game_state = "OVER"
                            self.set_message("GAME OVER!")
                            self.play_bgm("lose")
                            
                    self.enemy_attacks.remove(attack)
            
            # ターン交代処理
            attack_limit = 5 * len(self.enemies) if self.enemies else 0
            if self.enemy_attack_count >= attack_limit and not self.enemy_attacks:
                if not self.enemies:
                    self.game_state = "CLEAR"
                    self.set_message("GAME CLEAR!")
                    self.play_bgm("clear")
                else:
                    self.set_turn("PLAYER_TURN")
                    self.spawn_enemies(3)

    def handle_player_attack(self, target_pos):
        """プレイヤーの攻撃（クリック）処理"""
        player_gun_pos = (self.player.rect.centerx, self.player.rect.centery - 20)
        self.effects.append(BeamEffect(player_gun_pos, target_pos))
        if self.shoot_sound:
            self.shoot_sound.play()
        
        hit_found = False
        for enemy in self.enemies:
            if enemy.rect.collidepoint(target_pos):
                offset_x = target_pos[0] - enemy.rect.left
                offset_y = target_pos[1] - enemy.rect.top
                if enemy.image.get_at((int(offset_x), int(offset_y)))[3] > 0:
                    hit_found = True
                    points = calculate_score(target_pos, enemy)
                    self.score += points
                    self.floating_scores.add(FloatingScore(target_pos, points))
                    if self.hit_sound:
                        self.hit_sound.play()
                    if enemy.take_damage(25):
                        if not self.enemies:
                            self.game_state = "CLEAR"
                            self.set_message("GAME CLEAR!")
                            self.play_bgm("clear")
                    break
        if not hit_found:
            pass

    def draw(self):
        """すべての要素を画面に描画する"""
        pygame.display.flip()
        
        if self.background_image:
            self.screen.blit(self.background_image, (0, 0))
        else:
            self.screen.fill(BLACK)
            
        if self.game_state == "START":
            self.draw_text("Multi-Enemy Shooting Game", (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
            self.draw_text("PRESS SPACE TO START", (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            
        elif self.game_state == "PLAYING":
            self.player.draw(self.screen)
            for enemy in self.enemies:
                enemy.draw(self.screen)
            
            for effect in self.effects:
                effect.draw(self.screen)
            self.floating_scores.draw(self.screen)
            
            if self.turn == 'PLAYER_TURN':
                pygame.mouse.set_visible(True)
            elif self.turn == "ENEMY_TURN":
                pygame.mouse.set_visible(False)
                self.barrier.draw(self.screen)
                for attack in self.enemy_attacks:
                    progress = min(attack["timer"] / attack["duration"], 1.0)
                    current_radius = int(attack["max_radius"] * (1 - progress))
                    if current_radius > 0:
                        pygame.draw.circle(self.screen, RED, attack["pos"], current_radius, 5)

            # UIの描画
            self.draw_text(f"HP: {self.player.health}", (80, SCREEN_HEIGHT - 40))
            self.draw_text(f"TURN: {self.turn}", (SCREEN_WIDTH // 2, 20), color=YELLOW)
            self.draw_text(f"SCORE: {self.score}", (SCREEN_WIDTH // 2, 60))
            time_left = max(0, 10 - self.turn_timer)
            self.draw_text(f"TIME: {time_left:.1f}", (SCREEN_WIDTH - 120, 20))

        elif self.game_state in ("CLEAR", "OVER"):
            msg = "GAME CLEAR!" if self.game_state == "CLEAR" else "GAME OVER"
            color = GREEN if self.game_state == "CLEAR" else RED
            self.draw_text(msg, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3), color=color)
            self.draw_text("PRESS SPACE TO RESTART", (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            
        if self.message and pygame.time.get_ticks() - self.message_timer < 2000:
            self.draw_text(self.message, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))

    def set_turn(self, turn_name):
        """ターンを切り替える"""
        self.turn = turn_name
        self.turn_timer = 0
        self.set_message(turn_name)
        if turn_name == "ENEMY_TURN":
            self.enemy_attack_count = 0
            self.attack_spawn_timer = 1500

    def set_message(self, text):
        """画面中央に一時的なメッセージを表示する"""
        self.message = text
        self.message_timer = pygame.time.get_ticks()

    def draw_text(self, text, center_pos, color=WHITE):
        """指定された位置にテキストを描画する"""
        text_surf = self.font.render(text, True, color)
        text_rect = text_surf.get_rect(center=center_pos)
        self.screen.blit(text_surf, text_rect)

if __name__ == "__main__":
    game = Game()
    game.run()