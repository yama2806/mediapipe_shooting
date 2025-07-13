import pygame
import random
import math

# --- ゲームの基本設定 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
FPS = 60
FONT_SIZE = 36

# --- 色の定義 ---
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (50, 50, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
MAGENTA = (255, 0, 255)

# --- ゲームの挙動設定 ---
PLAYER_HP = 100
ENEMY_HP = 100
TURN_TIME = 15  # 各ターンの制限時間（秒）

# --- クラス定義 ---

class Target:
    """プレイヤーが撃つ単体の的を管理するクラス"""
    def __init__(self):
        self.base_radius = 60  # 的の基本サイズを大きく
        self.radius = self.base_radius
        self.pos = (random.randint(self.base_radius, SCREEN_WIDTH - self.base_radius),
                    random.randint(self.base_radius, SCREEN_HEIGHT - self.base_radius))
        self.rect = pygame.Rect(0, 0, self.radius * 2, self.radius * 2)
        self.rect.center = self.pos
        
        # 動きをダイナミックに
        self.move_direction = pygame.Vector2(random.choice([-1, 1]), random.choice([-1, 1])).normalize()
        self.speed = 5  # 移動速度をアップ
        self.scale = 1.0
        self.scale_direction = 1
        self.scale_speed = 0.02 # 拡大・縮小の速度

    def update(self):
        # 呼吸するように拡大・縮小
        self.scale += self.scale_direction * self.scale_speed
        if self.scale > 1.4 or self.scale < 0.6: # 1.4倍から0.6倍の間で変動
            self.scale_direction *= -1
        self.radius = self.base_radius * self.scale
        
        # 移動
        self.rect.move_ip(self.move_direction * self.speed)

        # 壁での反射
        if self.rect.left < 0 or self.rect.right > SCREEN_WIDTH:
            self.move_direction.x *= -1
        if self.rect.top < 0 or self.rect.bottom > SCREEN_HEIGHT:
            self.move_direction.y *= -1
        
        # self.posもrectの中心に追従させる
        self.pos = self.rect.center
        
    def draw(self, surface):
        # 現在の半径でRectを更新
        current_rect = pygame.Rect(0, 0, self.radius * 2, self.radius * 2)
        current_rect.center = self.pos
        
        pygame.draw.circle(surface, MAGENTA, self.pos, int(self.radius))
        pygame.draw.circle(surface, WHITE, self.pos, int(self.radius * 0.5), 3)
        pygame.draw.circle(surface, WHITE, self.pos, int(self.radius), 3)


class PlayerBulletEffect:
    """プレイヤーの弾の着弾エフェクト"""
    def __init__(self, pos):
        self.pos = pos
        self.radius = 10
        self.lifetime = 0.2 # 0.2秒で消える
        self.timer = 0.0

    def update(self, delta_time):
        self.timer += delta_time
        return self.timer < self.lifetime

    def draw(self, surface):
        alpha = 255 * (1 - self.timer / self.lifetime)
        effect_surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(effect_surface, (255, 255, 0, alpha), (self.radius, self.radius), self.radius)
        surface.blit(effect_surface, (self.pos[0] - self.radius, self.pos[1] - self.radius))


class IncomingAttack:
    """敵からの攻撃（着弾点）を管理するクラス"""
    def __init__(self):
        self.pos = (random.randint(100, SCREEN_WIDTH - 100), random.randint(100, SCREEN_HEIGHT - 100))
        self.max_radius = 120  # 予測円の初期サイズを大きく
        self.radius = self.max_radius
        self.duration = random.uniform(1.2, 2.0)  # 着弾までの時間を短く（収縮を速く）
        self.timer = 0.0
        self.is_active = True

    def update(self, delta_time):
        if not self.is_active: return
        self.timer += delta_time
        self.radius = self.max_radius * (1 - self.timer / self.duration)
        if self.timer >= self.duration:
            self.is_active = False

    def draw(self, surface):
        if self.is_active:
            pygame.draw.circle(surface, RED, self.pos, int(self.radius), 4)


class Shield:
    """プレイヤーの盾を管理するクラス"""
    def __init__(self):
        self.rect = pygame.Rect(SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT // 2 - 60, 120, 120)
        self.is_dragging = False

    def update(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.rect.collidepoint(event.pos):
                    self.is_dragging = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.is_dragging = False
        if self.is_dragging:
            self.rect.center = pygame.mouse.get_pos()

    def draw(self, surface):
        pygame.draw.ellipse(surface, GRAY, self.rect)
        pygame.draw.ellipse(surface, WHITE, self.rect, 4)


def main():
    """ゲームのメイン処理"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("一人称視点 ターン制シューティング")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, FONT_SIZE)
    big_font = pygame.font.Font(None, 100)

    player_hp = PLAYER_HP
    enemy_hp = ENEMY_HP
    score = 0
    game_state = 'change_turn'
    turn_text = "PLAYER TURN" # 最初にプレイヤーのターンから開始
    turn_timer = 2.0 # 最初だけすぐに始まるように

    target = None
    bullet_effects = []
    incoming_attacks = []
    shield = Shield()
    attack_spawn_timer = 0.0
    
    running = True
    while running:
        delta_time = clock.tick(FPS) / 1000.0
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False
            if game_state == 'player_turn' and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if target and pygame.Vector2(target.pos).distance_to(event.pos) < target.radius:
                    # 小さいほど高得点
                    score += int(100 * (1 / (target.radius / target.base_radius)))
                    enemy_hp -= 5
                    bullet_effects.append(PlayerBulletEffect(event.pos))
                    target = Target() # 新しい的を生成
                
        turn_timer += delta_time
        
        if game_state == 'player_turn':
            shield.is_dragging = False
            if target:
                target.update()
            if turn_timer > TURN_TIME:
                game_state = 'change_turn'
                turn_text = "ENEMY TURN"
                target = None

        elif game_state == 'enemy_turn':
            shield.update(events)
            attack_spawn_timer += delta_time
            if attack_spawn_timer > 1.5: # 1.5秒ごとに新しい攻撃
                attack_spawn_timer = 0
                incoming_attacks.append(IncomingAttack())
                
            for attack in incoming_attacks[:]:
                attack.update(delta_time)
                if not attack.is_active:
                    if shield.rect.collidepoint(attack.pos):
                        score += 10
                    else:
                        player_hp -= 10
                    incoming_attacks.remove(attack)
            
            if turn_timer > TURN_TIME:
                game_state = 'change_turn'
                turn_text = "PLAYER TURN"
                incoming_attacks.clear()
        
        elif game_state == 'change_turn':
            if turn_timer > 2.0:
                turn_timer = 0
                if turn_text == "PLAYER TURN":
                    game_state = 'player_turn'
                    target = Target()
                else:
                    game_state = 'enemy_turn'

        for effect in bullet_effects[:]:
            if not effect.update(delta_time):
                bullet_effects.remove(effect)

        if player_hp <= 0 or enemy_hp <= 0:
            running = False

        screen.fill(BLACK)
        
        if game_state == 'player_turn':
            if target:
                target.draw(screen)
            for effect in bullet_effects:
                effect.draw(screen)
        
        elif game_state == 'enemy_turn':
            for attack in incoming_attacks:
                attack.draw(screen)
            shield.draw(screen)
        
        elif game_state == 'change_turn':
            text_surface = big_font.render(turn_text, True, WHITE)
            screen.blit(text_surface, (SCREEN_WIDTH/2 - text_surface.get_width()/2, SCREEN_HEIGHT/2 - text_surface.get_height()/2))

        score_text = font.render(f"Score: {score}", True, WHITE)
        player_hp_text = font.render(f"Player HP: {player_hp}", True, GREEN)
        enemy_hp_text = font.render(f"Enemy HP: {enemy_hp}", True, RED)
        time_left = max(0, TURN_TIME - turn_timer)
        timer_text = font.render(f"Time: {time_left:.1f}", True, WHITE)

        screen.blit(score_text, (20, 20))
        screen.blit(player_hp_text, (20, 60))
        screen.blit(enemy_hp_text, (SCREEN_WIDTH - 220, 20))
        if game_state != 'change_turn':
            screen.blit(timer_text, (SCREEN_WIDTH/2 - timer_text.get_width()/2, 20))
        
        pygame.display.flip()

    game_over_font = pygame.font.Font(None, 80)
    if player_hp > 0:
        result_text = game_over_font.render("YOU WIN!", True, YELLOW)
    else:
        result_text = game_over_font.render("YOU LOSE...", True, RED)
    
    screen.blit(result_text, (SCREEN_WIDTH/2 - result_text.get_width()/2, SCREEN_HEIGHT/2 - result_text.get_height()/2))
    pygame.display.flip()
    pygame.time.wait(3000)

    pygame.quit()

if __name__ == "__main__":
    main()