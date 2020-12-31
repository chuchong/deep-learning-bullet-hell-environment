# 实现tabular的方法
from game.game_data import GameData
import numpy as np
class Tabular:
    def __init__(self, window_rect_w, window_rect_h):
        self.w = window_rect_w
        self.h = window_rect_h

    def table_size(self):
        return 64

    def get_state(self, game_data: GameData):
        region_bullets = np.zeros([4])

        center = game_data.player_point
        min_dist = 10000
        min_dist_id = 0
        for bullet in game_data.bullets:
            x = bullet.rect.center[0]
            y = bullet.rect.center[1]
            dist = abs(x - center[0]) + abs(y - center[1])
            cur = 0
            if x >= center[0] and y >= center[1]:
                cur = 0
            elif x < center[0] and y >= center[1]:
                cur = 1
            elif x < center[0]  and y < center[1]:
                cur = 2
            else:
                cur = 3
            region_bullets[cur] += 1
            if dist < min_dist:
                min_dist_id = cur
        # 加上密度
        s0 = (self.w - center[0])* center[1]
        s1 = center[0] * center[1]
        s2 = center[0] * (self.h - center[1])
        s3 = (self.w - center[0])* (self.h - center[1])
        region_dens = region_bullets / np.array([s0, s1, s2, s3])

        return 16 * np.argmax(region_bullets) + 4 * np.argmax(region_dens) + min_dist_id