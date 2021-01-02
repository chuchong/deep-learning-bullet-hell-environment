#将输入进行vectorize 化
from game.game_data import GameData
import numpy as np
from config import vectorize_d
class Vectorization:
    def __init__(self, window_rect_w, window_rect_h):
        self.w = window_rect_w
        self.h = window_rect_h
        self.d = vectorize_d

    def vector_size(self):
        return 2 + 4 + self.d * self.d

    def clamp(self, x, a, b):
        return x > a and x < b

    def get_vector(self, game_data: GameData):
        center = game_data.player_point
        vec = np.zeros([2 + self.d * self.d])
        vec[0] = center[0] / self.w
        vec[1] = center[1] / self.h

        for bullet in game_data.bullets:
            x = bullet.rect.center[0]
            y = bullet.rect.center[1]

            if self.clamp(x, 0, self.w) and self.clamp(y, 0, self.h):
                i = min(x // (self.w // self.d), self.d - 1)
                j = min(y // (self.h // self.d), self.d - 1)
                vec[2 + self.d * i + j] += 1
        # 加上密度
        vec[2:] = np.log2(vec[2:] + 1) + 0.01

        region_bullets = np.zeros([4])
        for bullet in game_data.bullets:
            x = bullet.rect.center[0]
            y = bullet.rect.center[1]
            dist = abs(x - center[0]) + abs(y - center[1])
            cur = 0
            if x >= center[0] and y >= center[1]:
                cur = 0
            elif x < center[0] and y >= center[1]:
                cur = 1
            elif x < center[0] and y < center[1]:
                cur = 2
            else:
                cur = 3
            region_bullets[cur] += 1
            # if dist < min_dist:
            #     min_dist_id = cur
            #     min_dist = dist
            # 加上密度
        s0 = (self.w - center[0]) * center[1]
        s1 = center[0] * center[1]
        s2 = center[0] * (self.h - center[1])
        s3 = (self.w - center[0]) * (self.h - center[1])
        region_dens = region_bullets / np.array([s0, s1, s2, s3])

        vec = np.hstack([vec, region_dens])
        return vec