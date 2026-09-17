#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agv_sim.py — 现代化智能仓储 AGV 3D 路径规划仿真 (Ursina + Panda3D)

真实算法：
  * 栅格 A* 寻路（八邻域 + octile 启发式 + 禁止穿墙角 + 障碍膨胀）
  * 视线（LOS）直线段路径平滑
  * 纯追踪 (pure pursuit) 差速运动学，真实角速度/线速度控制
  * 行进中周期性校验路径，被新增障碍阻断即自动重规划
  * 新增障碍后做连通性校验，会封死路径则回滚

交互：
  鼠标左键           当前模式下点击（目标 / 添加障碍 / 删除障碍，支持拖拽）
  鼠标右键拖拽       环绕视角；滚轮 缩放
  1/2/3/4 或按钮     切换模式：目标 / 货架 / 集装箱混合 / 删除
  F                  添加模式下旋转障碍朝向
  空格               暂停 / 继续；R 重新开始；N 随机新场景

  python3 agv_sim.py            正常运行
  python3 agv_sim.py --selftest 无头自检（随机场景 + A* + 模拟编辑，不弹窗）
  python3 agv_sim.py --smoke    弹窗冒烟测试（约 10 秒后自动退出）
"""

import sys
import os
import math
import random
import heapq
from collections import deque
from PIL import Image, ImageDraw

from ursina import (Entity, Ursina, color, Vec3, mouse, BoxCollider,
                    DirectionalLight, Cylinder, Text, Button, window,
                    held_keys, destroy, application, clamp, time as u_time,
                    camera)

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
GRID_W, GRID_H = 40, 30                 # 栅格数（每格 1m × 1m）
INFLATE = 1                             # 障碍膨胀格数（底盘安全余量）
ROBOT_SPEED = 2.4                       # m/s
ROBOT_OMEGA = 200.0                     # 最大角速度 度/秒
LOOKAHEAD = 1.0                         # 纯追踪前视距离 (m)
ARRIVE_DIST = 0.30
PATH_CHECK_INTERVAL = 0.4
TRAIL_SPACING = 0.14
TRAIL_MAX = 700

START_CELL = (3, GRID_H // 2)
GOAL_CELL = (GRID_W - 4, GRID_H // 2)

OB_TYPES = {
    'rack':      dict(name='货架',   footprint=(1, 3)),
    'container': dict(name='集装箱', footprint=(2, 1)),
    'equip':     dict(name='设备箱', footprint=(1, 1)),
    'fence':     dict(name='围栏',   footprint=(4, 1)),
}

CJK_FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',
]

_CJK_FONT = None

def cjk_font():
    """Ursina 的 Text.font 只认 ttf/otf 路径（不认 ttc），
    这里用 Panda3D DynamicTextFont 直接加载 Noto CJK ttc 后注入。"""
    global _CJK_FONT
    if _CJK_FONT is not None:
        return _CJK_FONT
    from panda3d.core import DynamicTextFont
    for p in CJK_FONT_CANDIDATES:
        if os.path.exists(p):
            _CJK_FONT = DynamicTextFont(p)
            return _CJK_FONT
    return None

def apply_cjk(text_entity, content=None):
    f = cjk_font()
    if f is not None:
        text_entity._font = f
        if content is not None:
            text_entity.text = content
    return text_entity

# ----------------------------------------------------------------------
# 过程化纹理
# ----------------------------------------------------------------------
def make_grid_texture():
    """浅灰仓储地面：浅灰底 + 细网格 + 每 5 格粗线。"""
    from ursina import Texture
    cell = 48
    img = Image.new('RGB', (GRID_W * cell, GRID_H * cell), (228, 231, 235))
    d = ImageDraw.Draw(img)
    fine = (206, 210, 216)
    bold = (180, 186, 194)
    W, H = GRID_W * cell, GRID_H * cell
    for i in range(GRID_W + 1):
        x = i * cell
        d.line([(x, 0), (x, H)], fill=bold if i % 5 == 0 else fine,
               width=3 if i % 5 == 0 else 1)
    for j in range(GRID_H + 1):
        y = j * cell
        d.line([(0, y), (W, y)], fill=bold if j % 5 == 0 else fine,
               width=3 if j % 5 == 0 else 1)
    t = Texture(img)
    t.filtering = None
    return t


# ----------------------------------------------------------------------
# 规划器：膨胀占据栅格 + A* + LOS 平滑
# ----------------------------------------------------------------------
class GridPlanner:
    def __init__(self, w=GRID_W, h=GRID_H, inflate=INFLATE):
        self.w, self.h = w, h
        self.inflate = inflate
        self.base = [[False] * h for _ in range(w)]
        self.blocked = [[True] * h for _ in range(w)]
        self._rebuild_inflation()

    def in_bounds(self, x, z):
        return 0 <= x < self.w and 0 <= z < self.h

    def add_base_cells(self, cells):
        for (x, z) in cells:
            if self.in_bounds(x, z):
                self.base[x][z] = True
        self._rebuild_inflation()

    def remove_base_cells(self, cells):
        for (x, z) in cells:
            if self.in_bounds(x, z):
                self.base[x][z] = False
        self._rebuild_inflation()

    def _rebuild_inflation(self):
        r, w, h = self.inflate, self.w, self.h
        for x in range(w):
            col = self.blocked[x]
            for z in range(h):
                blocked = False
                x0, x1 = max(0, x - r), min(w, x + r + 1)
                z0, z1 = max(0, z - r), min(h, z + r + 1)
                for nx in range(x0, x1):
                    ncol = self.base[nx]
                    for nz in range(z0, z1):
                        if ncol[nz]:
                            blocked = True
                            break
                    if blocked:
                        break
                col[z] = blocked

    def cell_free(self, cell):
        x, z = cell
        return self.in_bounds(x, z) and not self.blocked[x][z]

    @staticmethod
    def cell_to_world(cell):
        return Vec3(cell[0] - GRID_W / 2 + 0.5, 0, cell[1] - GRID_H / 2 + 0.5)

    @staticmethod
    def world_to_cell(pos):
        return (int(math.floor(pos.x + GRID_W / 2)),
                int(math.floor(pos.z + GRID_H / 2)))

    def clamp_goal(self, cell):
        """若目标格被占，BFS 找最近的空闲格。"""
        if self.cell_free(cell):
            return cell
        q = deque([cell])
        seen = {cell}
        while q:
            c = q.popleft()
            if self.cell_free(c):
                return c
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + dx, c[1] + dz)
                if self.in_bounds(*n) and n not in seen:
                    seen.add(n)
                    q.append(n)
        return None

    def astar(self, start, goal):
        if not self.in_bounds(*start) or not self.in_bounds(*goal):
            return []
        if start == goal:
            return [start]
        if self.blocked[goal[0]][goal[1]]:
            return []

        D2 = math.sqrt(2.0)

        def h(c):
            dx, dz = abs(c[0] - goal[0]), abs(c[1] - goal[1])
            return dx + dz + (D2 - 2) * min(dx, dz)

        open_heap = [(h(start), 0.0, start)]
        gscore = {start: 0.0}
        parent = {}
        closed = set()

        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal:
                path = [cur]
                while cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            closed.add(cur)
            cx, cz = cur
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dz == 0:
                        continue
                    nx, nz = cx + dx, cz + dz
                    if not self.in_bounds(nx, nz) or self.blocked[nx][nz]:
                        continue
                    if dx != 0 and dz != 0:
                        if self.blocked[cx][nz] or self.blocked[nx][cz]:
                            continue
                    nb = (nx, nz)
                    if nb in closed:
                        continue
                    step = D2 if dx and dz else 1.0
                    ng = g + step
                    if ng < gscore.get(nb, 1e18):
                        gscore[nb] = ng
                        parent[nb] = cur
                        heapq.heappush(open_heap, (ng + h(nb), ng, nb))
        return []

    def segment_blocked(self, a, b, skip_start=0.0):
        """世界坐标直线段是否与膨胀障碍相交（0.25m 采样）；
        skip_start 内的近端点不检查（机器人转弯时可能短暂贴近膨胀格）。"""
        dx, dz = b.x - a.x, b.z - a.z
        dist = math.hypot(dx, dz)
        if dist < 1e-6:
            return False
        steps = max(2, int(dist / 0.25))
        for i in range(steps + 1):
            t = i / steps
            if t * dist < skip_start:
                continue
            cx = int(math.floor(a.x + dx * t + GRID_W / 2))
            cz = int(math.floor(a.z + dz * t + GRID_H / 2))
            if not self.in_bounds(cx, cz) or self.blocked[cx][cz]:
                return True
        return False

    def smooth_path(self, wp):
        if len(wp) <= 2:
            return wp
        out = [wp[0]]
        i = 0
        n = len(wp)
        while i < n - 1:
            j = n - 1
            while j > i + 1:
                if not self.segment_blocked(wp[i], wp[j]):
                    break
                j -= 1
            out.append(wp[j])
            i = j
        return out

    def plan_world(self, start_pos, goal_pos):
        s = self.world_to_cell(start_pos)
        g = self.clamp_goal(self.world_to_cell(goal_pos))
        if g is None:
            return []
        cells = self.astar(s, g)
        if not cells:
            return []
        wp = [self.cell_to_world(c) for c in cells]
        wp[0] = Vec3(start_pos.x, 0, start_pos.z)
        wp[-1] = Vec3(goal_pos.x, 0, goal_pos.z)
        return self.smooth_path(wp)


# ----------------------------------------------------------------------
# 障碍工具
# ----------------------------------------------------------------------
def footprint_cells(cx, cz, nx, nz, rot):
    if rot == 90:
        nx, nz = nz, nx
    x0 = cx - nx // 2
    x1 = cx + (nx - 1) // 2
    z0 = cz - nz // 2
    z1 = cz + (nz - 1) // 2
    return [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]


def build_obstacle(kind, center, rot=0):
    """返回带整体 box collider 的实体；细节都是子网格。"""
    g = Entity(position=Vec3(center[0], 0, center[1]), rotation_y=rot)

    if kind == 'rack':  # 横梁式货架 1×3，3 层货位
        L, Wd, H = 2.7, 0.78, 2.6
        rack_c = color.hsv(210, 0.18, 0.55)
        beam_c = color.hsv(210, 0.28, 0.40)
        for s in (-1, 1):
            for sz in (-1, 1):
                Entity(parent=g, model='cube', color=rack_c,
                       position=(s * L / 2, 0.07, sz * Wd / 2),
                       scale=(0.12, H, 0.12))
        for lvl in (0.45, 1.35, 2.25):
            for sz in (-1, 1):
                Entity(parent=g, model='cube', color=beam_c,
                       position=(0, lvl, sz * Wd / 2), scale=(L, 0.12, 0.12))
            Entity(parent=g, model='cube', color=color.hsv(210, .1, .72),
                   position=(0, lvl + 0.12, 0), scale=(L - 0.25, 0.07, Wd - 0.18))
            for i, bx in enumerate((-0.85, 0.0, 0.85)):
                cc = (color.azure, color.orange, color.lime)[i]
                Entity(parent=g, model='cube', color=cc,
                       position=(bx, lvl + 0.34, (i - 1) * 0.12),
                       scale=(0.62, 0.36, 0.5))
        g.collider = BoxCollider(g, size=(L, H, Wd), center=(0, H / 2, 0))

    elif kind == 'container':  # 2×1 海运集装箱
        L, Wd, H = 1.85, 1.7, 1.35
        con_c = random.choice([color.hsv(24, .65, .62),
                               color.hsv(15, .6, .55),
                               color.hsv(190, .5, .5),
                               color.hsv(95, .4, .45)])
        Entity(parent=g, model='cube', color=con_c,
               position=(0, H / 2, 0), scale=(L, H, Wd))
        rib = color.hsv(0, 0, 0.35)
        for xx in (-.75, -.45, -.15, .15, .45, .75):
            for side in (-1, 1):
                Entity(parent=g, model='cube', color=rib,
                       position=(xx, H / 2, side * Wd / 2),
                       scale=(0.05, H * .96, 0.04))
        for yy in (.12, H - .12):
            for side in (-1, 1):
                Entity(parent=g, model='cube', color=rib,
                       position=(0, yy, side * Wd / 2), scale=(L, .08, .05))
        Entity(parent=g, model='cube', color=color.hsv(45, .25, .3),
               position=(-L / 2 - .01, H / 2, 0), scale=(.05, H * .9, Wd * .9))
        g.collider = BoxCollider(g, size=(L, H, Wd), center=(0, H / 2, 0))

    elif kind == 'equip':  # 设备箱 / 充电桩 1×1
        S, H = 0.8, 1.1
        Entity(parent=g, model='cube', color=color.hsv(48, .35, .62),
               position=(0, H / 2, 0), scale=(S, H, S))
        Entity(parent=g, model='cube', color=color.hsv(48, .2, .8),
               position=(0, H + 0.02, 0), scale=(S * 1.05, 0.05, S * 1.05))
        for i in range(3):
            Entity(parent=g, model='cube', color=color.hsv(0, 0, .25),
                   position=(0, .35 + i * .14, S / 2 + .01),
                   scale=(S * .55, .04, .03))
        Entity(parent=g, model='sphere', color=color.red,
               position=(S / 2 - .12, H - .15, S / 2 - .12), scale=0.1)
        g.collider = BoxCollider(g, size=(S, H + .1, S), center=(0, (H + .1) / 2, 0))

    elif kind == 'fence':  # 4×1 安全围栏
        L, H = 3.8, 1.15
        rail_c = color.hsv(212, .14, .62)
        for s in (-1, 1):
            for sz in (-1, 1):
                Entity(parent=g, model=Cylinder(8), color=rail_c,
                       position=(s * L / 2, H / 2, sz * .18),
                       scale=(.05, H, .05))
        for sz in (-1, 1):
            for yy in (H * .35, H * .75):
                Entity(parent=g, model='cube', color=rail_c,
                       position=(0, yy, sz * .18), scale=(L, .05, .05))
        Entity(parent=g, model='cube', color=color.hsv(45, .7, .55),
               position=(0, .12, 0), scale=(L, .08, .42))
        g.collider = BoxCollider(g, size=(L, 1.3, .46), center=(0, .65, 0))

    return g


# ----------------------------------------------------------------------
# AGV 模型
# ----------------------------------------------------------------------
def build_robot():
    root = Entity()
    body_c = color.hsv(205, .42, .55)
    dark = color.hsv(210, .25, .28)

    Entity(parent=root, model='cube', color=dark,
           position=(0, .22, 0), scale=(.92, .16, .66))
    Entity(parent=root, model='cube', color=body_c,
           position=(0, .37, 0), scale=(.76, .26, .54))
    Entity(parent=root, model='cube', color=color.hsv(205, .2, .72),
           position=(0, .52, 0), scale=(.6, .06, .42))
    Entity(parent=root, model='cube', color=color.hsv(45, .7, .55),
           position=(0, .3, .285), scale=(.5, .08, .03))

    wheels = []
    for sx in (-1, 1):
        for sz in (-1, 1):
            piv = Entity(parent=root, position=(sx * .33, .13, sz * .24))
            w = Entity(parent=piv, model=Cylinder(20), rotation=(90, 0, 0),
                       color=color.hsv(0, 0, .15), scale=(.145, .12, .145))
            Entity(parent=piv, model=Cylinder(20), rotation=(90, 0, 0),
                   color=color.hsv(0, 0, .75), scale=(.06, .13, .06))
            wheels.append(w)

    lidar = Entity(parent=root, position=(0, .56, 0))
    Entity(parent=lidar, model=Cylinder(24), color=color.hsv(0, 0, .18),
           scale=(.11, .09, .11))
    scan_bar = Entity(parent=lidar, model='cube',
                      color=color.rgba32(0, 220, 255, 130),
                      position=(.1, .07, 0), scale=(.22, .02, .03))

    beacon = Entity(parent=root, model='sphere', color=color.green,
                    position=(-.22, .58, .16), scale=.085)
    Entity(parent=root, model='sphere', color=color.rgba32(255, 240, 190, 255),
           position=(.3, .39, .28), scale=.05)

    root.wheels = wheels
    root.scan_bar = scan_bar
    root.beacon = beacon
    return root


# ----------------------------------------------------------------------
# 主仿真（Entity：update/input 由 Ursina 自动回调）
# ----------------------------------------------------------------------
class AGVSimulation(Entity):
    def __init__(self, smoke=False, shot=False):
        super().__init__()
        self.smoke = smoke
        self.shot = shot
        self.smoke_frames = 0
        self.planner = GridPlanner()
        self.obstacles = {}
        self._next_ob_id = 1

        self.mode = 'target'
        self.add_kind = 'rack'
        self.add_rot = 0
        self.paused = False
        self.replan_count = 0
        self.odom = 0.0
        self.message_text = ''
        self.message_until = 0.0

        self.start_pos = self.planner.cell_to_world(START_CELL)
        self.goal_pos = self.planner.cell_to_world(GOAL_CELL)
        self.robot_pos = Vec3(self.start_pos.x, 0, self.start_pos.z)
        self.robot_heading = 0.0
        self.robot_state = '规划中'
        self.current_speed = 0.0
        self.path = []
        self._need_replan = False
        self._check_acc = 0.0
        self._paint_cd = 0.0
        self._dragging = {'left': False, 'right': False}
        self._left_down_pos = (0, 0)
        self._left_moved = False

        # ---- 场景 ----
        self.floor = Entity(model='plane', texture=make_grid_texture(),
                            scale=(GRID_W, 1, GRID_H), collider='box')
        DirectionalLight(color=color.hsv(210, .12, .95), y=20, x=14, z=12,
                         shadows=True, rotation=(50, -35, 35))
        Entity(model='sphere', scale=400, color=color.hsv(212, .22, .92),
               unlit=True)

        self._build_perimeter_fence()
        self.randomize_scene(initial=True)

        self.robot = build_robot()
        self.robot.position = self.robot_pos

        self.goal_marker = self._make_goal_marker()
        Entity(model='cube', color=color.rgba32(80, 230, 140, 160),
               position=Vec3(self.start_pos.x, .015, self.start_pos.z),
               scale=(.85, .02, .85))

        # 路径 / 轨迹对象池
        self.path_markers = [Entity(model='cube', enabled=False,
                                    color=color.rgba32(0, 200, 255, 200),
                                    scale=(.22, .03, .22))
                             for _ in range(420)]
        self.trail_markers = [Entity(model='cube', enabled=False,
                                     color=color.rgba32(255, 150, 40, 220),
                                     scale=(.12, .02, .12))
                              for _ in range(TRAIL_MAX)]
        self.trail_ring = deque(maxlen=TRAIL_MAX)
        self._trail_acc = 0.0

        # 放置预览（池）
        self.preview = Entity(enabled=False, unlit=True)
        self.preview_tiles = [Entity(parent=self.preview, model='cube',
                                     unlit=True, enabled=False,
                                     scale=(.96, .03, .96))
                              for _ in range(16)]

        # ---- 相机参数 ----
        self._cam_dist = 25.0
        self._orbit_yaw = 0.0
        self._orbit_pitch = 50.0

        # ---- UI ----
        self.fkw = dict()
        self._build_ui()

        self._replan(count=False)
        self.set_mode('target')
        self.flash('点击地面设置新目标点；可用按钮添加/删除障碍')

    # ---------------- 场景 ----------------
    def _build_perimeter_fence(self):
        rail_c = color.hsv(212, .14, .6)
        hw, hh = GRID_W / 2, GRID_H / 2
        for yy in (.45, .95):
            Entity(model='cube', color=rail_c, position=(-.5, yy, -hh),
                   scale=(GRID_W, .06, .06))
            Entity(model='cube', color=rail_c, position=(-.5, yy, hh - 1),
                   scale=(GRID_W, .06, .06))
            Entity(model='cube', color=rail_c, position=(-hw, yy, -.5),
                   scale=(.06, .06, GRID_H))
            Entity(model='cube', color=rail_c, position=(hw - 1, yy, -.5),
                   scale=(.06, .06, GRID_H))
        for x in range(0, GRID_W, 2):
            Entity(model=Cylinder(8), color=rail_c,
                   position=Vec3(-hw + x, .6, -hh), scale=(.07, 1.25, .07))
            Entity(model=Cylinder(8), color=rail_c,
                   position=Vec3(-hw + x, .6, hh - 1), scale=(.07, 1.25, .07))
        for z in range(0, GRID_H, 2):
            Entity(model=Cylinder(8), color=rail_c,
                   position=Vec3(-hw, .6, -hh + z), scale=(.07, 1.25, .07))
            Entity(model=Cylinder(8), color=rail_c,
                   position=Vec3(hw - 1, .6, -hh + z), scale=(.07, 1.25, .07))

    def _make_goal_marker(self):
        m = Entity(position=self.goal_pos)
        Entity(parent=m, model=Cylinder(28), color=color.rgba32(255, 60, 60, 150),
               scale=(.55, .04, .55), position=(0, .03, 0))
        beam = Entity(parent=m, model=Cylinder(24),
                      color=color.rgba32(255, 70, 70, 60),
                      scale=(.12, 2.2, .12), position=(0, 1.1, 0))
        m.beam = beam
        return m

    def _clear_obstacles(self):
        for ob in self.obstacles.values():
            destroy(ob['entity'])
        self.obstacles.clear()
        self._next_ob_id = 1
        self.planner = GridPlanner()

    def randomize_scene(self, initial=False):
        if not initial:
            self._clear_obstacles()
            self.odom = 0.0
            self.replan_count = 0
            self.trail_ring.clear()
            self._trail_acc = 0.0
            for m in self.trail_markers:
                m.enabled = False

        def try_add(kind, cx, cz, rot):
            nx, nz = OB_TYPES[kind]['footprint']
            cells = footprint_cells(cx, cz, nx, nz, rot)
            if any(not self.planner.in_bounds(x, z) for x, z in cells):
                return False
            if any(self.planner.base[x][z] for x, z in cells):
                return False
            self.planner.add_base_cells(cells)
            if not self.planner.plan_world(self.start_pos, self.goal_pos):
                self.planner.remove_base_cells(cells)
                return False
            center = (cx - GRID_W / 2 + .5, cz - GRID_H / 2 + .5)
            ent = build_obstacle(kind, center, rot)
            ent.ob_id = self._next_ob_id
            self.obstacles[self._next_ob_id] = dict(kind=kind, cells=cells,
                                                    entity=ent, rot=rot,
                                                    center=(cx, cz))
            self._next_ob_id += 1
            return True

        # 两排整齐货架，形成主巷道
        for rz in (8, 21):
            for cx in range(6, GRID_W - 6, 5):
                try_add('rack', cx, rz + random.choice((-1, 0, 1)), 0)
        for _ in range(4):
            try_add('rack', random.randrange(5, GRID_W - 5),
                    random.randrange(4, GRID_H - 4), random.choice((0, 90)))
        for _ in range(7):
            for _ in range(12):
                if try_add('container', random.randrange(3, GRID_W - 3),
                           random.randrange(3, GRID_H - 3),
                           random.choice((0, 90))):
                    break
        for _ in range(9):
            for _ in range(12):
                if try_add('equip', random.randrange(2, GRID_W - 2),
                           random.randrange(2, GRID_H - 2), 0):
                    break
        for _ in range(5):
            for _ in range(15):
                if try_add('fence', random.randrange(4, GRID_W - 4),
                           random.randrange(3, GRID_H - 3),
                           random.choice((0, 90))):
                    break

        if not initial:
            self.robot_pos = Vec3(self.start_pos.x, 0, self.start_pos.z)
            self.robot_heading = 0.0
            self.robot.position = self.robot_pos
            self.robot.rotation_y = 0
            self.goal_pos = self.planner.cell_to_world(GOAL_CELL)
            self.goal_marker.position = self.goal_pos
            self.paused = False
            self._replan(count=False)
            self.flash('已随机生成新仓储场景')

    # ---------------- 障碍编辑 ----------------
    def add_obstacle_at(self, kind, cell, rot, quiet=False):
        cx, cz = cell
        nx, nz = OB_TYPES[kind]['footprint']
        cells = footprint_cells(cx, cz, nx, nz, rot)
        if any(not self.planner.in_bounds(x, z) for x, z in cells):
            return False
        if any(self.planner.base[x][z] for x, z in cells):
            return False
        protect = (self.planner.world_to_cell(self.robot_pos),
                   self.planner.world_to_cell(self.start_pos),
                   self.planner.world_to_cell(self.goal_pos))
        if set(cells) & set(protect):
            if not quiet:
                self.flash('不能在机器人 / 起点 / 目标上放置障碍')
            return False

        self.planner.add_base_cells(cells)
        new_path = self.planner.plan_world(self.robot_pos, self.goal_pos)
        if not new_path:
            self.planner.remove_base_cells(cells)
            if not quiet:
                self.flash('放置后无可达路径，已自动取消')
            return False

        center = (cx - GRID_W / 2 + .5, cz - GRID_H / 2 + .5)
        ent = build_obstacle(kind, center, rot)
        ent.ob_id = self._next_ob_id
        self.obstacles[self._next_ob_id] = dict(kind=kind, cells=cells,
                                                entity=ent, rot=rot,
                                                center=(cx, cz))
        self._next_ob_id += 1
        self.path = new_path
        self.replan_count += 1
        self.refresh_path_markers()
        if not quiet:
            self.flash(f'已添加{OB_TYPES[kind]["name"]}，自动重新规划')
        return True

    def remove_obstacle(self, oid, quiet=False):
        ob = self.obstacles.pop(oid, None)
        if ob is None:
            return False
        self.planner.remove_base_cells(ob['cells'])
        destroy(ob['entity'])
        new_path = self.planner.plan_world(self.robot_pos, self.goal_pos)
        if new_path:
            self.path = new_path
            self.replan_count += 1
            self.refresh_path_markers()
        if not quiet:
            self.flash(f'已删除{OB_TYPES[ob["kind"]]["name"]}')
        return True

    def set_goal(self, world_pos):
        cell = self.planner.world_to_cell(world_pos)
        if not self.planner.in_bounds(*cell):
            return
        target = self.planner.clamp_goal(cell)
        if target is None:
            self.flash('附近没有可通行区域')
            return
        wp = self.planner.cell_to_world(target)
        if math.hypot(wp.x - self.robot_pos.x, wp.z - self.robot_pos.z) < .8:
            self.flash('目标点太靠近机器人')
            return
        self.goal_pos = wp
        self.goal_marker.position = wp
        self._replan()
        self.flash('目标点已更新，重新规划路径')

    # ---------------- 规划 ----------------
    def _replan(self, count=True):
        new_path = self.planner.plan_world(self.robot_pos, self.goal_pos)
        if new_path:
            self.path = new_path
            if count:
                self.replan_count += 1
            self.refresh_path_markers()
            self._need_replan = False
        else:
            self._need_replan = True
            self.path = []
            self.refresh_path_markers()
        return new_path

    def refresh_path_markers(self):
        idx = 0
        for wp in self.path[1:]:
            if idx >= len(self.path_markers):
                break
            m = self.path_markers[idx]
            m.enabled = True
            m.position = Vec3(wp.x, .035, wp.z)
            idx += 1
        for m in self.path_markers[idx:]:
            m.enabled = False

    def path_ahead_blocked(self):
        """沿规划路径本身检查前方走廊是否被新增障碍阻断；
        机器人偏离路径超过 0.7m（被新障碍挤离）也触发重规划。"""
        if not self.path:
            return True
        best_seg, best_t, best_d2 = self._nearest_path_info()
        # 偏离阈值留足切角余量；且刚重规划（投影点在路径起点）时不重复触发
        if best_d2 > 1.0 ** 2:
            return True
        a, b = self.path[best_seg], self.path[best_seg + 1]
        proj = Vec3(a.x + (b.x - a.x) * best_t, 0, a.z + (b.z - a.z) * best_t)
        chain = [proj] + self.path[best_seg + 1:best_seg + 8]
        for i in range(len(chain) - 1):
            if self.planner.segment_blocked(chain[i], chain[i + 1]):
                return True
        return False

    # ---------------- 运动学：纯追踪差速控制 ----------------
    def update_robot(self, dt):
        dist_goal = math.hypot(self.goal_pos.x - self.robot_pos.x,
                               self.goal_pos.z - self.robot_pos.z)
        if dist_goal < ARRIVE_DIST:
            self.robot_state = '已到达'
            self.current_speed = 0.0
            return
        if not self.path:
            self.robot_state = '受阻·重规划中'
            self.current_speed = 0.0
            return

        # ---- 标准纯追踪：投影到最近路径段，再沿路径取前视点 ----
        seg_i, seg_t, dist2 = self._nearest_path_info()
        # 已被抛在身后的航点弹出
        while seg_i > 0 and self.path:
            self.path.pop(0)
            seg_i -= 1

        a, b = self.path[seg_i], self.path[seg_i + 1]
        proj = Vec3(a.x + (b.x - a.x) * seg_t, 0, a.z + (b.z - a.z) * seg_t)
        # 从投影点沿路径累计 LOOKAHEAD
        target = self._lookahead_point(seg_i, seg_t, LOOKAHEAD)

        dx, dz = target.x - self.robot_pos.x, target.z - self.robot_pos.z
        ang = (math.degrees(math.atan2(dx, dz)) - self.robot_heading + 180) % 360 - 180

        if abs(ang) > 100:
            self.robot_state = '原地转向'
            v = 0.0
        elif abs(ang) > 18:
            self.robot_state = '转向行进'
            v = ROBOT_SPEED * max(.25, 1 - abs(ang) / 110)
        else:
            self.robot_state = '自主行进'
            v = ROBOT_SPEED

        self.robot_heading += clamp(ang, -ROBOT_OMEGA * dt, ROBOT_OMEGA * dt)

        if v > 0 and abs(ang) < 75:
            step = v * dt
            hd = math.radians(self.robot_heading)
            self.robot_pos.x += math.sin(hd) * step
            self.robot_pos.z += math.cos(hd) * step
            self.odom += step
            self.current_speed = v
            for w in self.robot.wheels:
                w.rotation_x -= math.degrees(step / .145)
            self._trail_acc += step
            if self._trail_acc >= TRAIL_SPACING:
                self._add_trail()
                self._trail_acc = 0.0
            self.refresh_path_markers()
        else:
            self.current_speed = 0.0

        self.robot.position = self.robot_pos
        self.robot.rotation_y = self.robot_heading

    def _nearest_path_info(self):
        """返回 (最近段索引, 段内参数 t, 平方距离)。"""
        rx, rz = self.robot_pos.x, self.robot_pos.z
        best = (0, 0.0, 1e18)
        for i in range(len(self.path) - 1):
            a, b = self.path[i], self.path[i + 1]
            dx, dz = b.x - a.x, b.z - a.z
            l2 = dx * dx + dz * dz
            t = 0.0 if l2 < 1e-9 else clamp(((rx - a.x) * dx + (rz - a.z) * dz) / l2, 0, 1)
            d2 = (a.x + dx * t - rx) ** 2 + (a.z + dz * t - rz) ** 2
            if d2 < best[2]:
                best = (i, t, d2)
        return best

    def _lookahead_point(self, seg_i, seg_t, lookahead):
        """从 (seg_i, seg_t) 沿路径累计 lookahead 米，插值返回前视点。"""
        a, b = self.path[seg_i], self.path[seg_i + 1]
        remain = (1 - seg_t) * math.hypot(b.x - a.x, b.z - a.z)
        if remain >= lookahead:
            tt = seg_t + lookahead / max(math.hypot(b.x - a.x, b.z - a.z), 1e-9)
            return Vec3(a.x + (b.x - a.x) * tt, 0, a.z + (b.z - a.z) * tt)
        need = lookahead - remain
        j = seg_i + 1
        while j < len(self.path) - 1:
            seg_len = math.hypot(self.path[j + 1].x - self.path[j].x,
                                 self.path[j + 1].z - self.path[j].z)
            if seg_len >= need:
                t = need / max(seg_len, 1e-9)
                return Vec3(self.path[j].x + (self.path[j + 1].x - self.path[j].x) * t,
                            0,
                            self.path[j].z + (self.path[j + 1].z - self.path[j].z) * t)
            need -= seg_len
            j += 1
        return Vec3(self.path[-1].x, 0, self.path[-1].z)

    def _add_trail(self):
        m = self.trail_markers[len(self.trail_ring) % TRAIL_MAX]
        m.enabled = True
        m.position = Vec3(self.robot_pos.x, .025, self.robot_pos.z)
        self.trail_ring.append(m)

    # ---------------- 鼠标拾取 ----------------
    def _mouse_over_ui(self):
        ent = mouse.hovered_entity
        while ent is not None:
            if isinstance(ent, Button):
                return True
            ent = ent.parent
        return False

    def _ground_hit(self):
        """返回 (world_point, obstacle_root_entity) 或 (None, None)。"""
        if mouse.hovered_entity is None or mouse.world_point is None:
            return None, None
        ent = mouse.hovered_entity
        p = mouse.world_point
        if ent == self.floor:
            return Vec3(p.x, 0, p.z), None
        root = ent
        while root is not None and not hasattr(root, 'ob_id'):
            root = root.parent
        if root is not None:
            return Vec3(p.x, 0, p.z), root
        return None, None

    def handle_click(self, point, ob):
        if self.mode == 'target':
            if ob is not None:
                self.flash('请点击空闲地面设置目标点')
            else:
                self.set_goal(point)
        elif self.mode in ('add_rack', 'add_mix'):
            self.add_obstacle_at(self.add_kind,
                                 self.planner.world_to_cell(point), self.add_rot)
        else:  # delete
            if ob is not None:
                self.remove_obstacle(ob.ob_id)
            else:
                self.flash('请点击要删除的障碍物')

    def handle_paint(self, dt):
        self._paint_cd -= dt
        if self._paint_cd > 0 or not self._dragging['left']:
            return
        if self.mode == 'target' or self._mouse_over_ui():
            return
        p, ob = self._ground_hit()
        if p is None:
            return
        if self.mode == 'delete':
            if ob is not None:
                self.remove_obstacle(ob.ob_id, quiet=True)
                self._paint_cd = .15
        else:
            if self.add_obstacle_at(self.add_kind,
                                    self.planner.world_to_cell(p),
                                    self.add_rot, quiet=True):
                self._paint_cd = .22

    def _update_preview(self):
        tiles = self.preview_tiles
        if self.mode not in ('add_rack', 'add_mix') or self._mouse_over_ui():
            self.preview.enabled = False
            return
        p, _ = self._ground_hit()
        if p is None:
            self.preview.enabled = False
            return
        cx, cz = self.planner.world_to_cell(p)
        nx, nz = OB_TYPES[self.add_kind]['footprint']
        cells = footprint_cells(cx, cz, nx, nz, self.add_rot)
        ok = (not any(not self.planner.in_bounds(x, z) for x, z in cells)
              and not any(self.planner.base[x][z] for x, z in cells))
        col = color.rgba32(90, 230, 120, 110) if ok else color.rgba32(240, 80, 80, 110)
        for i, t in enumerate(tiles):
            if i < len(cells):
                x, z = cells[i]
                t.enabled = True
                t.position = Vec3(x - GRID_W / 2 + .5, .02, z - GRID_H / 2 + .5)
                t.color = col
            else:
                t.enabled = False
        self.preview.enabled = True

    # ---------------- UI ----------------
    def _build_ui(self):
        # 顶部 / 底部半透明条
        Entity(parent=camera.ui, model='quad',
               color=color.rgba32(15, 25, 40, 210),
               position=(0, .462, 1), scale=(1.06, .085))
        Entity(parent=camera.ui, model='quad',
               color=color.rgba32(15, 25, 40, 210),
               position=(0, -.462, 1), scale=(1.06, .075))
        # HUD 底板（左上）
        Entity(parent=camera.ui, model='quad',
               color=color.rgba32(15, 25, 40, 190),
               position=(-.635, .33, 1), scale=(.30, .235))
        self.hud = apply_cjk(Text(
            text='',
            position=window.top_left + Vec3(.02, -.125, -1),
            scale=.78), '')

        btns = [
            ('target',  '目标[1]'),
            ('add_rack', '货架[2]'),
            ('add_mix',  '集装箱/设备/围栏[3]'),
            ('delete',   '删除[4]'),
        ]
        self.mode_buttons = {}
        widths = (.09, .09, .205, .09)
        gap = .012
        total = sum(widths) + gap * (len(widths) - 1)
        x = -total / 2
        for (m, label), w in zip(btns, widths):
            b = Button(parent=camera.ui, text=label,
                       position=(x + w / 2, .462),
                       scale=(w, .055), color=color.hsv(210, .3, .35, .95),
                       text_color=color.white)
            apply_cjk(b.text_entity, label)
            b.text_entity.scale *= .82
            if m == 'add_mix':
                b.on_click = self._cycle_mix
            else:
                b.on_click = lambda m=m: self.set_mode(m)
            self.mode_buttons[m] = b
            x += w + gap

        actions = [
            ('rotate',  'F 旋转'),
            ('pause',   '空格 暂停'),
            ('restart', 'R 重置'),
            ('random',  'N 随机场景'),
        ]
        widths2 = (.09, .11, .10, .13)
        total2 = sum(widths2) + gap * 3
        x = -total2 / 2
        for (act, label), w in zip(actions, widths2):
            b = Button(parent=camera.ui, text=label,
                       position=(x + w / 2, -.462),
                       scale=(w, .05), color=color.hsv(220, .25, .3, .95),
                       text_color=color.white)
            apply_cjk(b.text_entity, label)
            b.text_entity.scale *= .82
            b.on_click = lambda act=act: self.do_action(act)
            x += w + gap

        apply_cjk(Text(text='',
                       position=window.bottom_right + Vec3(-.012, -.448, -1),
                       origin=(.7, 0), scale=.62, color=color.hsv(210, .1, .82)),
                  '左键：当前模式操作（可拖拽）　右键拖拽：旋转视角　滚轮：缩放')
        self.msg_text = apply_cjk(Text(text='',
                                       position=window.top_right + Vec3(-.02, -.11, -1),
                                       origin=(.7, .5), scale=.85,
                                       color=color.yellow), '')

    def do_action(self, act):
        if act == 'rotate':
            self.add_rot = 0 if self.add_rot == 90 else 90
            self.flash(f'障碍朝向: {self.add_rot}°')
        elif act == 'pause':
            self.paused = not self.paused
            self.flash('已暂停' if self.paused else '继续运行')
        elif act == 'restart':
            self.restart()
        elif act == 'random':
            self.randomize_scene()

    def _cycle_mix(self):
        cycle = ['container', 'equip', 'fence']
        if self.mode != 'add_mix' or self.add_kind not in cycle:
            self.set_mode('add_mix')
            self.add_kind = 'container'
        else:
            self.add_kind = cycle[(cycle.index(self.add_kind) + 1) % 3]
            self.flash(f'添加类型: {OB_TYPES[self.add_kind]["name"]}')

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'add_mix' and self.add_kind not in ('container', 'equip', 'fence'):
            self.add_kind = 'container'
        elif mode == 'add_rack':
            self.add_kind = 'rack'
        for m, b in self.mode_buttons.items():
            b.color = (color.hsv(190, .7, .5, .95) if m == mode
                       else color.hsv(210, .3, .35, .92))

    def flash(self, msg):
        self.message_text = msg
        self.message_until = u_time.time() + 2.6

    def restart(self):
        self.robot_pos = Vec3(self.start_pos.x, 0, self.start_pos.z)
        self.robot_heading = 0.0
        self.robot.position = self.robot_pos
        self.robot.rotation_y = 0
        self.odom = 0.0
        self.replan_count = 0
        self.trail_ring.clear()
        for m in self.trail_markers:
            m.enabled = False
        self.paused = False
        self._replan(count=False)
        self.flash('机器人已重置到起点')

    # ---------------- 相机 ----------------
    def _update_camera(self):
        t = Vec3(self.robot_pos.x, 0, self.robot_pos.z)
        yaw, pitch = math.radians(self._orbit_yaw), math.radians(self._orbit_pitch)
        d = self._cam_dist
        camera.world_position = (
            t.x + d * math.cos(pitch) * math.sin(yaw),
            t.y + d * math.sin(pitch) + 1.0,
            t.z + d * math.cos(pitch) * math.cos(yaw),
        )
        camera.look_at(t + Vec3(0, .4, 0))

    # ---------------- HUD ----------------
    def _update_hud(self):
        path_len = 0.0
        if self.path:
            chain = [Vec3(self.robot_pos.x, 0, self.robot_pos.z)] + self.path
            for i in range(len(chain) - 1):
                path_len += math.hypot(chain[i + 1].x - chain[i].x,
                                       chain[i + 1].z - chain[i].z)
        state_cn = {'已到达': '已到达 ✓', '受阻·重规划中': '受阻·重规划中 !'}.get(
            self.robot_state, self.robot_state)
        pause_cn = '已暂停' if self.paused else '运行中'
        kind_name = OB_TYPES.get(self.add_kind, {}).get('name', '')
        mode_cn = {'target': '设置目标点', 'delete': '删除障碍',
                   'add_rack': f'添加货架（朝向 {self.add_rot}°）',
                   'add_mix': f'添加{kind_name}（朝向 {self.add_rot}°）'}[self.mode]
        self.hud.text = (
            '仓储 AGV 路径规划仿真\n'
            f'状态: {state_cn}    [{pause_cn}]\n'
            f'模式: {mode_cn}\n'
            f'规划路径长度: {path_len:6.2f} m\n'
            f'剩余距离:     {path_len:6.2f} m\n'
            f'行驶里程:     {self.odom:6.2f} m\n'
            f'当前速度:     {self.current_speed:5.2f} m/s\n'
            f'重新规划次数: {self.replan_count}'
        )
        self.msg_text.text = (self.message_text
                              if u_time.time() < self.message_until else '')

    # ---------------- Ursina 回调 ----------------
    def update(self):
        dt = clamp(u_time.dt, .0001, .05)
        self._update_camera()

        pulse = .85 + .25 * math.sin(u_time.time() * 5)
        self.goal_marker.beam.color = color.rgba32(255, int(70 * pulse),
                                                 int(70 * pulse), 60)
        self.robot.scan_bar.rotation_y = (u_time.time() * 240) % 360

        if not self.paused:
            self._check_acc += dt
            if self._check_acc >= PATH_CHECK_INTERVAL:
                self._check_acc = 0.0
                if self.path_ahead_blocked():
                    self._replan()
            self.update_robot(dt)

        st = self.robot_state
        bc = {'已到达': color.lime,
              '受阻·重规划中': color.orange,
              '规划中': color.azure}.get(st, color.green)
        blink = st in ('受阻·重规划中', '规划中') and math.sin(u_time.time() * 12) > 0
        self.robot.beacon.color = color.dark_gray if blink else bc

        self.handle_paint(dt)
        self._update_preview()
        self._update_hud()
        if self.smoke:
            self.smoke_frames += 1
            if self.smoke_frames == 150:
                self.set_mode('add_mix')
                self.add_kind = 'fence'
            elif self.smoke_frames == 400:
                self.randomize_scene()
            elif self.smoke_frames > 750:
                print('SMOKE_OK')
                application.quit()
        if self.shot:
            self.smoke_frames += 1
            if self.smoke_frames in (120, 300):
                i = 0 if self.smoke_frames == 120 else 1
                self._app.screenshot(
                    os.path.abspath(f'agv_shot{i}.png'), defaultFilename=False)
                print(f'SHOT{i}_OK')
            if self.smoke_frames > 360:
                application.quit()

    def input(self, key):
        if key == 'left mouse down':
            self._dragging['left'] = True
            self._left_down_pos = (mouse.position[0], mouse.position[1])
            self._left_moved = False
        elif key == 'left mouse up':
            was_drag = self._dragging['left']
            self._dragging['left'] = False
            if was_drag and not self._left_moved and not self._mouse_over_ui():
                p, ob = self._ground_hit()
                if p is not None:
                    self.handle_click(p, ob)
        elif key == 'right mouse down':
            self._dragging['right'] = True
        elif key == 'right mouse up':
            self._dragging['right'] = False
        elif key == 'mouse moved':
            if self._dragging['left']:
                x0, y0 = self._left_down_pos
                if math.hypot(mouse.position[0] - x0,
                              mouse.position[1] - y0) > .012:
                    self._left_moved = True
            if self._dragging['right']:
                v = mouse.velocity
                self._orbit_yaw += v[0] * 90
                self._orbit_pitch = clamp(self._orbit_pitch - v[1] * 70, 22, 82)
        elif key == 'scroll up':
            self._cam_dist = clamp(self._cam_dist - 2.2, 7, 55)
        elif key == 'scroll down':
            self._cam_dist = clamp(self._cam_dist + 2.2, 7, 55)
        elif key == '1':
            self.set_mode('target')
        elif key == '2':
            self.set_mode('add_rack')
        elif key == '3':
            # 重复按 3 在集装箱/设备箱/围栏之间循环
            cycle = ['container', 'equip', 'fence']
            if self.mode != 'add_mix' or self.add_kind not in cycle:
                self.set_mode('add_mix')
                self.add_kind = 'container'
            else:
                self.add_kind = cycle[(cycle.index(self.add_kind) + 1) % 3]
                self.flash(f'添加类型: {OB_TYPES[self.add_kind]["name"]}')
        elif key == '4':
            self.set_mode('delete')
        elif key == 'f':
            self.do_action('rotate')
        elif key == 'space':
            self.do_action('pause')
        elif key == 'r':
            self.restart()
        elif key == 'n':
            self.randomize_scene()


# ----------------------------------------------------------------------
# 无头自检
# ----------------------------------------------------------------------
def selftest():
    random.seed(42)
    print('=== AGV 仿真无头自检 ===')
    planner = GridPlanner()
    start = planner.cell_to_world(START_CELL)
    goal = planner.cell_to_world(GOAL_CELL)

    # 1) 空场景 A*
    path = planner.plan_world(start, goal)
    assert path, '空场景必须能规划出路径'
    L0 = sum(math.hypot(path[i+1].x - path[i].x, path[i+1].z - path[i].z)
             for i in range(len(path) - 1))
    print(f'[1] 空场景 A*: {len(path)} 个航点, 长度 {L0:.2f} m')

    # 2) 横墙留缺口（膨胀半径 1，缺口至少 3 格宽）
    wall_x = (START_CELL[0] + GOAL_CELL[0]) // 2
    gap = (11, 12, 13)
    for z in range(GRID_H):
        if z not in gap:
            planner.add_base_cells([(wall_x, z)])
    path2 = planner.plan_world(start, goal)
    assert path2, '留缺口的墙必须可通行'
    near = [wp for wp in path2
            if 10.0 <= wp.z + GRID_H / 2 <= 14.0
            and abs(wp.x - (wall_x - GRID_W / 2 + .5)) <= 1.0]
    assert near, '路径必须从缺口附近通过'
    print(f'[2] 绕墙路径: {len(path2)} 航点, 从缺口 z=11~13 通过 ✓')

    # 3) 完全封死 -> 无路径
    for gz in gap:
        planner.add_base_cells([(wall_x, gz)])
    assert not planner.plan_world(start, goal), '完全封墙应无路径'
    for gz in gap:
        planner.remove_base_cells([(wall_x, gz)])
    assert planner.plan_world(start, goal), '解封后应恢复'
    print('[3] 完全封死无路径 / 解封恢复 ✓')

    # 4) 膨胀 3×3
    planner2 = GridPlanner(inflate=1)
    planner2.add_base_cells([(20, 15)])
    assert planner2.blocked[20][15] and planner2.blocked[21][15] \
        and planner2.blocked[20][16] and planner2.blocked[19][14]
    assert not planner2.base[21][15]
    print('[4] 障碍膨胀 3×3 ✓')

    # 5) 随机场景连通性（模拟 GUI 的生成 + 连通性校验策略）
    for seed in range(20):
        random.seed(seed)
        p3 = GridPlanner()

        def place(kind, cx, cz, rot):
            nx, nz = OB_TYPES[kind]['footprint']
            cells = footprint_cells(cx, cz, nx, nz, rot)
            if any(not p3.in_bounds(x, z) for x, z in cells):
                return False
            if any(p3.base[x][z] for x, z in cells):
                return False
            p3.add_base_cells(cells)
            if not p3.plan_world(start, goal):
                p3.remove_base_cells(cells)
                return False
            return True

        nobs = 0
        for rz in (8, 21):
            for cx in range(6, GRID_W - 6, 5):
                if place('rack', cx, rz + random.choice((-1, 0, 1)), 0):
                    nobs += 1
        for _ in range(30):
            k = random.choice(['container', 'equip', 'fence'])
            if place(k, random.randrange(3, GRID_W - 3),
                     random.randrange(3, GRID_H - 3), random.choice((0, 90))):
                nobs += 1
        final = p3.plan_world(start, goal)
        assert final and nobs >= 15, f'seed {seed}: 场景无效 nobs={nobs}'
    print('[5] 20 个随机场景全部连通且障碍数充足 ✓')

    # 6) 纯追踪收敛
    pos = Vec3(start.x, 0, start.z)
    heading = 0.0
    dt = 1 / 60
    for _ in range(60 * 30):
        ang = math.degrees(math.atan2(goal.x - pos.x, goal.z - pos.z)) - heading
        ang = (ang + 180) % 360 - 180
        heading += clamp(ang, -ROBOT_OMEGA * dt, ROBOT_OMEGA * dt)
        if abs(ang) < 75:
            pos.x += math.sin(math.radians(heading)) * ROBOT_SPEED * dt
            pos.z += math.cos(math.radians(heading)) * ROBOT_SPEED * dt
        if math.hypot(goal.x - pos.x, goal.z - pos.z) < ARRIVE_DIST:
            break
    err = math.hypot(goal.x - pos.x, goal.z - pos.z)
    assert err < ARRIVE_DIST + .05, f'纯追踪未收敛: err={err:.2f}'
    print(f'[6] 纯追踪运动学收敛, 终点误差 {err:.3f} m ✓')

    print('=== 全部自检通过 ===')


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main():
    if '--selftest' in sys.argv:
        selftest()
        return

    # 限制帧率：无 vsync 环境下避免突发渲染导致的画面卡顿
    from panda3d.core import loadPrcFileData
    loadPrcFileData('', 'sync-video 0')
    loadPrcFileData('', 'clock-mode limited')
    loadPrcFileData('', 'clock-frame-rate 60')
    if '--windowed' in sys.argv:
        loadPrcFileData('', 'win-size 1400 900')
        loadPrcFileData('', 'fullscreen 0')

    smoke = '--smoke' in sys.argv
    shot = '--shot' in sys.argv
    app = Ursina(title='仓储 AGV 3D 路径规划仿真', borderless=False,
                 development_mode=False, fullscreen='--windowed' not in sys.argv)
    if '--windowed' in sys.argv:
        window.borderless = False
    window.color = color.hsv(212, .22, .92)
    window.fps_counter.enabled = False
    window.exit_button.visible = False

    sim = AGVSimulation(smoke=smoke, shot=shot)
    sim._app = app
    app.run()


if __name__ == '__main__':
    main()
