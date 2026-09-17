"""
真实路径规划核心模块（与渲染引擎无关，可独立单元测试）

实现内容：
1. 将连续场景中的 AABB 障碍物栅格化到均匀网格（含机器人半径膨胀 + 围栏边界）
2. 8 邻接 A* 搜索（octile 距离作为一致启发函数，禁止贴角穿越）
3. 贪心视线（line-of-sight）路径平滑 —— "拉线" 去拐点
4. 最近可行点搜索（BFS），用于目标点落在障碍物内时自动吸附

不存在任何固定路线或预设航点：每次调用 plan() 都根据当前障碍物实时搜索。
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Vec2 = Tuple[float, float]  # (x, z) 世界坐标


@dataclass
class BoxObstacle:
    """轴对齐矩形障碍物（俯视图）。y 高度与规划无关。"""
    x: float
    z: float
    w: float   # x 方向宽度
    d: float   # z 方向深度

    def inflate(self, amount: float) -> "BoxObstacle":
        return BoxObstacle(self.x, self.z, self.w + 2 * amount, self.d + 2 * amount)


class GridWorld:
    def __init__(self, width: float = 40.0, depth: float = 30.0,
                 cell_size: float = 0.5, inflation: float = 0.45,
                 border: float = 0.75):
        self.width = width
        self.depth = depth
        self.cell = cell_size
        self.inflation = inflation
        self.border = border
        self.nx = int(round(width / cell_size))
        self.nz = int(round(depth / cell_size))
        # 1 表示阻塞
        self.blocked = bytearray(self.nx * self.nz)
        # 供调试/可视化用的最近一次栅格化的膨胀矩形
        self.raster_rects: List[Tuple[int, int, int, int]] = []

    # ---------- 坐标转换 ----------
    def world_to_cell(self, x: float, z: float) -> Tuple[int, int]:
        ix = int((x + self.width / 2.0) / self.cell)
        iz = int((z + self.depth / 2.0) / self.cell)
        return ix, iz

    def cell_to_world(self, ix: int, iz: int) -> Vec2:
        x = -self.width / 2.0 + (ix + 0.5) * self.cell
        z = -self.depth / 2.0 + (iz + 0.5) * self.cell
        return x, z

    def in_bounds(self, ix: int, iz: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iz < self.nz

    def is_free_cell(self, ix: int, iz: int) -> bool:
        return self.in_bounds(ix, iz) and not self.blocked[iz * self.nx + ix]

    def is_free_world(self, x: float, z: float) -> bool:
        ix, iz = self.world_to_cell(x, z)
        return self.is_free_cell(ix, iz)

    # ---------- 栅格化 ----------
    def rasterize(self, obstacles: Iterable[BoxObstacle]) -> None:
        """根据当前障碍物集合重建阻塞图。"""
        self.blocked = bytearray(self.nx * self.nz)
        self.raster_rects = []

        # 围栏 / 场地边界：留出 border 宽的安全带
        bcells = int(math.ceil(self.border / self.cell))
        for ix in range(self.nx):
            for b in range(bcells):
                self.blocked[b * self.nx + ix] = 1
                self.blocked[(self.nz - 1 - b) * self.nx + ix] = 1
        for iz in range(self.nz):
            for b in range(bcells):
                self.blocked[iz * self.nx + b] = 1
                self.blocked[iz * self.nx + (self.nx - 1 - b)] = 1

        # 障碍物（按机器人外接半径膨胀，保证路径中心不会贴边穿过）
        for ob in obstacles:
            exp = ob.inflate(self.inflation)
            x0, x1 = exp.x - exp.w / 2.0, exp.x + exp.w / 2.0
            z0, z1 = exp.z - exp.d / 2.0, exp.z + exp.d / 2.0
            ix0 = max(0, int(math.floor((x0 + self.width / 2.0) / self.cell)))
            ix1 = min(self.nx - 1, int(math.floor((x1 + self.width / 2.0) / self.cell)))
            iz0 = max(0, int(math.floor((z0 + self.depth / 2.0) / self.cell)))
            iz1 = min(self.nz - 1, int(math.floor((z1 + self.depth / 2.0) / self.cell)))
            if ix0 > ix1 or iz0 > iz1:
                continue
            self.raster_rects.append((ix0, iz0, ix1, iz1))
            for iz in range(iz0, iz1 + 1):
                base = iz * self.nx
                for ix in range(ix0, ix1 + 1):
                    self.blocked[base + ix] = 1

    # ---------- 最近可行点 ----------
    def nearest_free_cell(self, ix: int, iz: int,
                          max_radius: Optional[int] = None) -> Optional[Tuple[int, int]]:
        """从 (ix,iz) 出发做 BFS，返回最近的空闲栅格；目标本身可行时原样返回。"""
        if max_radius is None:
            max_radius = max(self.nx, self.nz)
        if self.in_bounds(ix, iz):
            if not self.blocked[iz * self.nx + ix]:
                return ix, iz
        else:
            ix = min(max(ix, 0), self.nx - 1)
            iz = min(max(iz, 0), self.nz - 1)
        visited = bytearray(self.nx * self.nz)
        visited[iz * self.nx + ix] = 1
        q = deque([(ix, iz, 0)])
        while q:
            cx, cz, r = q.popleft()
            if r >= max_radius:
                return None
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, nz = cx + dx, cz + dz
                if not self.in_bounds(nx, nz):
                    continue
                idx = nz * self.nx + nx
                if visited[idx]:
                    continue
                visited[idx] = 1
                if not self.blocked[idx]:
                    return nx, nz
                q.append((nx, nz, r + 1))
        return None

    # ---------- A* ----------
    def astar(self, start_cell: Tuple[int, int], goal_cell: Tuple[int, int]
              ) -> Optional[List[Tuple[int, int]]]:
        (sx, sz), (gx, gz) = start_cell, goal_cell
        if not (self.is_free_cell(sx, sz) and self.is_free_cell(gx, gz)):
            return None
        if start_cell == goal_cell:
            return [start_cell]

        SQRT2 = math.sqrt(2.0)

        def h(ix: int, iz: int) -> float:
            dx, dz = abs(ix - gx), abs(iz - gz)
            return (max(dx, dz) - min(dx, dz)) + SQRT2 * min(dx, dz)

        start_idx = sz * self.nx + sx
        goal_idx = gz * self.nx + gx
        g_score = {start_idx: 0.0}
        came: dict = {}
        open_heap = [(h(sx, sz), 0.0, start_idx)]
        closed = bytearray(self.nx * self.nz)
        counter = 1

        while open_heap:
            _, _, cur_idx = heapq.heappop(open_heap)
            if closed[cur_idx]:
                continue
            if cur_idx == goal_idx:
                path = []
                i = cur_idx
                while i != start_idx:
                    path.append((i % self.nx, i // self.nx))
                    i = came[i]
                path.append((sx, sz))
                path.reverse()
                return path
            closed[cur_idx] = 1
            cx, cz = cur_idx % self.nx, cur_idx // self.nx
            cur_g = g_score[cur_idx]

            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dz == 0:
                        continue
                    nx, nz = cx + dx, cz + dz
                    if not self.in_bounds(nx, nz):
                        continue
                    n_idx = nz * self.nx + nx
                    if closed[n_idx] or self.blocked[n_idx]:
                        continue
                    diagonal = dx != 0 and dz != 0
                    if diagonal:
                        # 禁止贴角：斜走要求两个正交邻居都空闲，避免从夹缝边角挤过去
                        if self.blocked[cz * self.nx + nx] or \
                           self.blocked[nz * self.nx + cx]:
                            continue
                    step = SQRT2 if diagonal else 1.0
                    tentative = cur_g + step
                    if tentative < g_score.get(n_idx, math.inf):
                        g_score[n_idx] = tentative
                        came[n_idx] = cur_idx
                        heapq.heappush(open_heap,
                                       (tentative + h(nx, nz), counter, n_idx))
                        counter += 1
        return None

    # ---------- 视线（用于平滑与碰撞检测） ----------
    def segment_blocked(self, p0: Vec2, p1: Vec2) -> bool:
        """线段是否经过任何阻塞栅格。

        使用 Amanatides-Woo 2D DDA 精确枚举线段穿过的每一个栅格，
        不依赖稀疏采样，避免短距离擦过障碍角时漏检。
        """
        x0, z0 = p0
        x1, z1 = p1
        ix, iz = self.world_to_cell(x0, z0)
        if not self.is_free_cell(ix, iz):
            return True
        ex, ez = self.world_to_cell(x1, z1)
        if not self.in_bounds(ex, ez) or self.blocked[ez * self.nx + ex]:
            return True
        if (ix, iz) == (ex, ez):
            return False

        dx, dz = x1 - x0, z1 - z0
        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_z = 1 if dz > 0 else (-1 if dz < 0 else 0)

        def next_boundary(i: int, positive: bool) -> float:
            # 栅格 i 的世界坐标边界（正方向取上边，负方向取下边）
            edge = -self.width / 2.0 + (i + 1) * self.cell if positive \
                else -self.width / 2.0 + i * self.cell
            return edge

        eps = 1e-9
        if step_x != 0:
            bx = next_boundary(ix, step_x == 1)
            t_max_x = (bx - x0) / (dx or eps)
            t_delta_x = self.cell / abs(dx)
        else:
            t_max_x = math.inf
            t_delta_x = math.inf
        if step_z != 0:
            bz = -self.depth / 2.0 + (iz + 1) * self.cell if step_z == 1 \
                else -self.depth / 2.0 + iz * self.cell
            t_max_z = (bz - z0) / (dz or eps)
            t_delta_z = self.cell / abs(dz)
        else:
            t_max_z = math.inf
            t_delta_z = math.inf

        cx, cz = ix, iz
        # 最多遍历两个轴上的栅格总数，杜绝数值问题导致死循环
        for _ in range(self.nx + self.nz + 4):
            if (cx, cz) == (ex, ez):
                return False
            if t_max_x < t_max_z:
                cx += step_x
                t_max_x += t_delta_x
            else:
                cz += step_z
                t_max_z += t_delta_z
            if not self.in_bounds(cx, cz) or self.blocked[cz * self.nx + cx]:
                return True
        return True

    def smooth(self, path: Sequence[Vec2]) -> List[Vec2]:
        """贪心拉线平滑：从当前点尽量连到最远的可见路径点。"""
        if len(path) <= 2:
            return list(path)
        result = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if not self.segment_blocked(path[i], path[j]):
                    break
                j -= 1
            result.append(path[j])
            i = j
        return result


def path_length(path: Sequence[Vec2]) -> float:
    return sum(math.hypot(path[i + 1][0] - path[i][0],
                          path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1))


def plan_path(world: GridWorld, start: Vec2, goal: Vec2
              ) -> Tuple[Optional[List[Vec2]], Optional[Tuple[int, int]]]:
    """
    完整规划流程：
    1. 起终点吸附到最近空闲栅格（目标点落进障碍物时由 BFS 自动找最近可行点）
    2. A* 搜索
    3. 用真实起终点坐标替换栅格端点并做视线平滑
    返回 (平滑后的世界坐标路径, 实际使用的目标栅格)；失败返回 (None, None)。
    """
    sx, sz = start
    gx, gz = goal
    sc = world.nearest_free_cell(*world.world_to_cell(sx, sz))
    gc = world.nearest_free_cell(*world.world_to_cell(gx, gz))
    if sc is None or gc is None:
        return None, None
    cells = world.astar(sc, gc)
    if cells is None:
        return None, gc
    pts = [world.cell_to_world(ix, iz) for ix, iz in cells]
    pts[0] = (sx, sz)
    pts[-1] = (gx, gz)
    return world.smooth(pts), gc
