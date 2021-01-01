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
        return 2 + self.d * self.d

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

        return vec