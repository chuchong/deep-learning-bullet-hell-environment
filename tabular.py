# 实现tabular的方法
from game.game_data import GameData
import numpy as np
class Tabular:
    def __init__(self, window_rect_w, window_rect_h):
        self.w = window_rect_w
        self.h = window_rect_h


    def get_state(self, game_data: GameData):
        region_bullets = np.zeros([4])

        center = game_data.player_point
        for bullet in game_data.bullets:
            x = bullet.rect.center[0]
            y = bullet.rect.center[1]
            if x >= center[0] and y >= center[1]:
                region_bullets[0] += 1
            elif x < center[0] and y >= center[1]:
                region_bullets[1] += 1
            elif x < center[0]  and y < center[1]:
                region_bullets[2] += 1
            else:
                region_bullets[3] += 1
        return np.argmax(region_bullets)