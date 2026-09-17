"""
3D 智能仓储 AGV 路径规划仿真  (Ursina / Panda3D)

真实算法：
  - 障碍栅格化（按机器人外接半径膨胀）+ 8 邻接 A* + 视线平滑  —— planner.py
  - 鼠标增删障碍 / 重设目标点后，实时重建栅格并重新规划
  - 运行中持续对前方路径做碰撞采样，被新障碍阻断即自动重规划
  - 差速底盘运动学：限速、线加速度、最大角速度、车轮按 v/r 真实滚动
没有任何固定路线或预设动画。

操作：
  左键          依当前模式：设置目标点 / 添加障碍 / 删除障碍
  Tab / 1/2/3   切换 目标/添加/删除 模式
  Q             切换添加的障碍类型（货架/集装箱/围栏/设备箱）
  空格          暂停 / 继续； R 重新开始； N 随机场景； Esc 退出
  鼠标右键拖动  旋转视角； 滚轮 缩放； 中键 平移

无窗口自测： AGV_SMOKE=1 python3 agv_sim.py
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from panda3d.core import loadPrcFileData

# ---- 窗口配置（必须在 import ursina 之前） ----
if os.environ.get('AGV_SMOKE'):
    loadPrcFileData('', 'window-type offscreen')
loadPrcFileData('', 'window-title 智能仓储 AGV 路径规划仿真')
loadPrcFileData('', 'framebuffer-multisample 1')
loadPrcFileData('', 'multisamples 4')

from ursina import (  # noqa: E402
    Ursina, Entity, EditorCamera, DirectionalLight, AmbientLight,
    Button, Text, color, Vec2, application,
    mouse, destroy, window, camera, clamp, Cylinder,
)
from ursina import time as u_time  # noqa: E402

from planner import GridWorld, BoxObstacle, plan_path, path_length  # noqa: E402

# ============================== 参数 ==============================
ARENA_W, ARENA_D = 40.0, 30.0
CELL = 0.5
ROBOT_RADIUS = 0.45
MAX_SPEED = 3.2          # m/s
MAX_ANGULAR = 120.0      # deg/s
ACCEL = 2.4              # m/s^2
STOP_TOL = 0.55          # m（需大于中间路径点消费半径）
WP_TOL = 0.35            # m，中间路径点消费半径

START_POS = (-17.0, -11.0)
INIT_GOAL = (16.0, 11.0)

MODE_TARGET = '目标点'
MODE_ADD = '添加障碍'
MODE_DELETE = '删除障碍'

OB_KINDS = {
    '货架':   dict(w=3.2, d=1.1, h=3.6),
    '集装箱': dict(w=4.6, d=2.3, h=2.4),
    '围栏':   dict(w=3.6, d=0.35, h=1.05),
    '设备箱': dict(w=1.6, d=1.4, h=1.9),
}
KIND_ORDER = list(OB_KINDS)

C_SHELF = color.rgb32(70, 92, 130)
C_CONTAINER = color.rgb32(150, 102, 58)
C_FENCE = color.rgb32(214, 176, 52)
C_BOX = color.rgb32(96, 130, 116)


@dataclass
class Obstacle:
    name: str
    box: BoxObstacle
    entity: Entity


class AGVSimulation:
    def __init__(self) -> None:
        self.app = Ursina(borderless=False)
        window.color = color.rgb32(226, 230, 236)
        window.fps_counter.enabled = False
        try:
            Text.default_font = 'cjk.otf'
        except Exception:
            pass

        # ---- 灯光 ----
        dlight = DirectionalLight(shadows=True,
                                  color=color.rgb32(225, 228, 235),
                                  position=(10, 34, -16))
        dlight.look_at((0, 0, 0))
        AmbientLight(color=color.rgba32(150, 155, 165, 255))

        # ---- 地面与围栏边界 ----
        self._build_floor()

        # ---- 规划世界 ----
        self.world = GridWorld(ARENA_W, ARENA_D, CELL, inflation=0.5)
        self.obstacles: List[Obstacle] = []

        # ---- 机器人与标记 ----
        self.robot = self._build_robot()
        self.goal_pos: Optional[Vec2] = None
        self.goal_marker = self._build_goal_marker()
        self.path: List[Tuple[float, float]] = []
        self.path_entities: List[Entity] = []
        self.trail: List[Tuple[float, float]] = []
        self.trail_entities: List[Entity] = []

        # 运动状态
        self.speed = 0.0
        self.heading = 0.0
        self.paused = False
        self.status = '待机'
        self.replan_count = 0
        self.block_cd = 0.0

        # 交互
        self.mode = MODE_TARGET
        self.add_kind = KIND_ORDER[0]
        self.hover_point: Optional[Tuple[float, float]] = None
        self.ghost = Entity(model='cube', enabled=False, unlit=True,
                            origin_y=-0.5)

        # ---- 初始场景 ----
        self._build_initial_scene()
        self.rerasterize()
        self.clear_trail()
        self.set_goal(Vec2(*INIT_GOAL), count_replan=False)

        # ---- UI / 相机 ----
        # UI 延迟到首帧 aspect_ratio 稳定后再构建，避免被窗口的
        # 宽高比自动修正逻辑多次缩放坐标
        self._ui_ready = False
        self.hud = None
        self.kind_text = None
        self.mode_buttons = {}
        self.editor = EditorCamera(rotation=(55, -28, 0),
                                   position=(0, 26, -20), fov=60)

    # ========================== 场景 ==========================
    def _build_floor(self) -> None:
        tex_path = self._make_grid_texture()
        ground = Entity(model='plane', scale=(ARENA_W, 1, ARENA_D),
                        texture=tex_path,
                        texture_scale=(ARENA_W / 2.0, ARENA_D / 2.0),
                        color=color.rgb32(232, 234, 238), collider='box')
        ground.is_ground = True
        self.ground = ground
        # 场地边框
        for w_, d_, px, pz in (
            (ARENA_W + 2, 0.4, 0, -ARENA_D / 2 - 0.2),
            (ARENA_W + 2, 0.4, 0, ARENA_D / 2 + 0.2),
            (0.4, ARENA_D + 2, -ARENA_W / 2 - 0.2, 0),
            (0.4, ARENA_D + 2, ARENA_W / 2 + 0.2, 0),
        ):
            Entity(model='cube', scale=(w_, 0.12, d_),
                   position=(px, 0.06, pz),
                   color=color.rgb32(170, 178, 190))

    @staticmethod
    def _make_grid_texture() -> str:
        from PIL import Image
        s = 64
        img = Image.new('RGBA', (s, s), (236, 238, 242, 255))
        p = img.load()
        for i in range(s):
            p[i, 0] = (196, 200, 208, 255)
            p[0, i] = (196, 200, 208, 255)
            p[i, 1] = (214, 217, 223, 255)
            p[1, i] = (214, 217, 223, 255)
        path = str(application.asset_folder / '_grid_floor.png')
        img.save(path)
        return path

    def _make_obstacle_entity(self, name: str, x: float, z: float) -> Entity:
        if name == '货架':
            root = Entity(position=(x, 0, z))
            k = OB_KINDS['货架']
            for ox in (-k['w'] / 4, k['w'] / 4):
                for oz in (-k['d'] / 2 + 0.12, k['d'] / 2 - 0.12):
                    Entity(parent=root, model='cube',
                           scale=(0.14, k['h'], 0.14),
                           position=(ox, k['h'] / 2, oz),
                           color=color.rgb32(46, 58, 84), collider='box')
            for lev in (0.9, 1.9, 2.9):
                Entity(parent=root, model='cube',
                       scale=(k['w'], 0.12, k['d']),
                       position=(0, lev, 0),
                       color=color.rgb32(82, 106, 148), collider='box')
            Entity(parent=root, model='cube',
                   scale=(k['w'], 0.1, k['d']),
                   position=(0, k['h'], 0), color=C_SHELF)
            return root

        if name == '围栏':
            root = Entity(position=(x, 0, z))
            k = OB_KINDS['围栏']
            Entity(parent=root, model='cube',
                   scale=(k['w'], 0.08, 0.08),
                   position=(0, 0.55, 0), color=C_FENCE, collider='box')
            Entity(parent=root, model='cube',
                   scale=(k['w'], 0.08, 0.08),
                   position=(0, 0.95, 0), color=C_FENCE, collider='box')
            for ox in (-k['w'] / 2, k['w'] / 2):
                for oz in (-0.12, 0.12):
                    Entity(parent=root, model='cube',
                           scale=(0.1, 1.05, 0.1),
                           position=(ox, 0.52, oz),
                           color=color.rgb32(180, 140, 40), collider='box')
            return root

        k = OB_KINDS[name]
        c = C_CONTAINER if name == '集装箱' else C_BOX
        # 根实体不缩放；所有子件用真实尺寸，避免嵌套缩放累乘
        root = Entity(position=(x, 0, z))
        Entity(parent=root, model='cube',
               scale=(k['w'], k['h'], k['d']),
               position=(0, k['h'] / 2, 0), color=c, collider='box')
        if name == '集装箱':
            for fy in (-0.55, 0, 0.55):   # 相对箱体高度 -0.5..0.5
                Entity(parent=root, model='cube',
                       scale=(k['w'] * 1.02, 0.12, k['d'] * 1.02),
                       position=(0, k['h'] * (0.5 + fy), 0),
                       color=color.rgb32(118, 78, 44))
        else:
            # 前面板
            Entity(parent=root, model='cube',
                   scale=(k['w'] * 0.5, k['h'] * 0.32, 0.08),
                   position=(0, k['h'] * 0.35, k['d'] / 2 + 0.005),
                   color=color.rgb32(30, 34, 40))
            # 中部腰线
            Entity(parent=root, model='cube',
                   scale=(k['w'] * 1.02, 0.10, k['d'] * 1.02),
                   position=(0, k['h'] * 0.65, 0),
                   color=color.rgb32(70, 96, 86))
        return root

    def add_obstacle(self, name: str, x: float, z: float,
                     silent: bool = False) -> bool:
        k = OB_KINDS[name]
        if not self._can_place(x, z, k['w'], k['d']):
            return False
        ent = self._make_obstacle_entity(name, x, z)
        self.obstacles.append(Obstacle(name, BoxObstacle(x, z, k['w'], k['d']),
                                       ent))
        if not silent:
            self.rerasterize()
            if self.goal_pos is not None:
                self.do_replan()
        return True

    def remove_obstacle(self, root: Entity) -> bool:
        for i, ob in enumerate(self.obstacles):
            if ob.entity == root:
                destroy(ob.entity)
                del self.obstacles[i]
                self.rerasterize()
                if self.goal_pos is not None:
                    self.do_replan()
                return True
        return False

    def _can_place(self, x: float, z: float, w: float, d: float) -> bool:
        if abs(x) + w / 2 > ARENA_W / 2 - 0.6 or \
           abs(z) + d / 2 > ARENA_D / 2 - 0.6:
            return False
        for ob in self.obstacles:
            if (abs(x - ob.box.x) < (w + ob.box.w) / 2 + 0.15 and
                    abs(z - ob.box.z) < (d + ob.box.d) / 2 + 0.15):
                return False
        # 不得让障碍的膨胀栅格覆盖机器人脚下位置：
        # 保留 膨胀半径 + 一个栅格的量化余量
        clear = self.world.inflation + self.world.cell
        if abs(x - self.robot.x) < w / 2 + clear and \
           abs(z - self.robot.z) < d / 2 + clear:
            return False
        return True

    def _build_initial_scene(self) -> None:
        layout = [
            ('货架', -4, 2.0), ('货架', 1.0, 2.0), ('货架', 6.0, 2.0),
            ('货架', -1.5, 8.5), ('货架', 3.5, 8.5),
            ('集装箱', 9.5, -6.5), ('集装箱', -8.5, -5.5),
            ('围栏', -2.0, -4.0), ('围栏', 4.0, -8.5),
            ('设备箱', 11.5, 6.5), ('设备箱', -11.0, 7.0),
            ('设备箱', 7.5, 12.0),
        ]
        for name, x, z in layout:
            self.add_obstacle(name, float(x), float(z), silent=True)

    def random_scene(self) -> None:
        for ob in self.obstacles:
            destroy(ob.entity)
        self.obstacles.clear()
        rng = random.Random()
        target = rng.randint(14, 21)
        attempts = 0
        placed = 0
        reserved = (START_POS, INIT_GOAL)
        while placed < target and attempts < 800:
            attempts += 1
            name = rng.choices(KIND_ORDER, weights=(4, 2, 2, 2))[0]
            x = round(rng.uniform(-ARENA_W / 2 + 3, ARENA_W / 2 - 3), 1)
            z = round(rng.uniform(-ARENA_D / 2 + 3, ARENA_D / 2 - 3), 1)
            if any(math.hypot(x - rx, z - rz) < 4.5 for rx, rz in reserved):
                continue
            if self.add_obstacle(name, x, z, silent=True):
                placed += 1
        self.rerasterize()
        # 极端情况下起终点不通则随机再来一次
        test, _ = plan_path(self.world, START_POS, INIT_GOAL)
        if test is None:
            return self.random_scene()
        self.reset_robot(keep_goal=False)
        self.set_goal(Vec2(*INIT_GOAL), count_replan=False)

    def rerasterize(self) -> None:
        self.world.rasterize(ob.box for ob in self.obstacles)

    # ========================== 机器人 ==========================
    def _build_robot(self) -> Entity:
        root = Entity(position=(START_POS[0], 0, START_POS[1]))
        # 低矮底盘 + 上车体
        Entity(parent=root, model='cube', scale=(0.95, 0.22, 1.25),
               position=(0, 0.28, 0), color=color.rgb32(224, 158, 28))
        Entity(parent=root, model='cube', scale=(0.78, 0.20, 0.95),
               position=(0, 0.47, 0), color=color.rgb32(38, 44, 54))
        # 激光雷达转台（持续旋转）
        self.turret = Entity(parent=root, position=(0, 0.64, 0))
        Entity(parent=self.turret, model=Cylinder(resolution=16, start=-.5),
               scale=(0.30, 0.16, 0.30), color=color.rgb32(20, 24, 30))
        Entity(parent=self.turret, model='cube',
               scale=(0.07, 0.10, 0.34), position=(0, 0.07, 0.16),
               color=color.rgb32(40, 220, 220))
        Entity(parent=root, model=Cylinder(resolution=16, start=-.5), scale=(0.16, 0.10, 0.16),
               position=(0, 0.80, 0), color=color.rgb32(60, 70, 84))
        # 状态灯
        self.status_light = Entity(parent=root, model='sphere',
                                   scale=0.16, position=(0, 0.66, -0.36),
                                   color=color.cyan, unlit=True)
        # 四个可见车轮：枢轴（滚动角）-> 轮体（cylinder 默认轴向 Y，绕 Z 转 90°
        # 使车轴指向车体 X），枢轴绕 X 旋转即为车轮绕车轴真实滚动
        self.wheels = []
        for ox, oz in ((-0.52, 0.42), (0.52, 0.42),
                       (-0.52, -0.42), (0.52, -0.42)):
            pivot = Entity(parent=root, position=(ox, 0.20, oz))
            Entity(parent=pivot, model=Cylinder(resolution=16, start=-.5),
                   rotation=(0, 0, 90), scale=(0.40, 0.13, 0.40),
                   color=color.rgb32(28, 30, 34))
            self.wheels.append(pivot)
        # 前部车灯
        Entity(parent=root, model='cube', scale=(0.30, 0.06, 0.08),
               position=(0, 0.52, 0.60), color=color.rgb32(255, 240, 180))
        return root

    def _build_goal_marker(self) -> Entity:
        g = Entity(enabled=False)
        Entity(parent=g, model=Cylinder(resolution=16, start=-.5), scale=(0.05, 2.4, 0.05),
               position=(0, 1.2, 0), color=color.rgba32(80, 200, 120, 160))
        self.goal_ring = Entity(parent=g, model=Cylinder(resolution=16, start=-.5),
                                scale=(0.75, 0.05, 0.75),
                                position=(0, 0.05, 0),
                                color=color.rgba32(60, 220, 130, 150))
        self.goal_beacon = Entity(parent=g, model='sphere', scale=0.28,
                                  position=(0, 2.45, 0),
                                  color=color.rgb32(60, 220, 130), unlit=True)
        return g

    # ========================== 路径可视化 ==========================
    def clear_path_viz(self) -> None:
        for e in self.path_entities:
            destroy(e)
        self.path_entities = []

    def render_path(self) -> None:
        self.clear_path_viz()
        if len(self.path) < 2:
            return
        for a, b in zip(self.path, self.path[1:]):
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            yaw = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))
            e = Entity(model='cube',
                       scale=(0.12, 0.03, max(seg_len, 0.02)),
                       position=((a[0] + b[0]) / 2, 0.07, (a[1] + b[1]) / 2),
                       rotation=(0, yaw, 0),
                       color=color.rgb32(40, 170, 255), unlit=True)
            self.path_entities.append(e)
        for x, z in self.path:
            e = Entity(model='sphere', scale=0.13, position=(x, 0.1, z),
                       color=color.rgb32(120, 210, 255), unlit=True)
            self.path_entities.append(e)

    def add_trail_point(self, p: Tuple[float, float]) -> None:
        self.trail.append(p)
        e = Entity(model='cube', scale=(0.16, 0.04, 0.16),
                   position=(p[0], 0.03, p[1]),
                   color=color.rgb32(255, 80, 200), unlit=True)
        self.trail_entities.append(e)

    def clear_trail(self) -> None:
        for e in self.trail_entities:
            destroy(e)
        self.trail_entities = []
        self.trail = []

    # ========================== 规划 ==========================
    def set_goal(self, p: Vec2, count_replan: bool = False) -> None:
        self.goal_pos = Vec2(float(p.x), float(p.y))
        self.goal_marker.enabled = True
        self.goal_marker.position = (p.x, 0, p.y)
        self.clear_trail()
        self._recompute(count_replan=count_replan)

    def do_replan(self) -> None:
        self._recompute(count_replan=True)

    def _recompute(self, count_replan: bool) -> bool:
        if self.goal_pos is None:
            self.path = []
            self.render_path()
            return False
        new_path, _ = plan_path(
            self.world,
            (self.robot.x, self.robot.z),
            (self.goal_pos.x, self.goal_pos.y))
        if new_path is None:
            self.path = []
            self.status = '无可行路径'
            self.render_path()
            return False
        gx, gz = new_path[-1]
        # BFS 吸附：目标点实际被移动到最近可达点时同步标记
        if math.hypot(gx - self.goal_pos.x, gz - self.goal_pos.y) > 0.3:
            self.goal_pos = Vec2(gx, gz)
            self.goal_marker.position = (gx, 0, gz)
        self.path = new_path
        if count_replan:
            self.replan_count += 1
        self.render_path()
        return True

    def remaining_distance(self) -> float:
        if not self.path:
            return 0.0
        d = math.hypot(self.robot.x - self.path[0][0],
                       self.robot.z - self.path[0][1])
        return d + path_length(self.path)

    # ========================== 交互 ==========================
    def _ui_hovered(self) -> bool:
        e = mouse.hovered_entity
        while e is not None:
            if getattr(e, '_is_ui', False):
                return True
            e = e.parent
        return False

    def _world_hover(self) -> Optional[Tuple[float, float]]:
        """返回鼠标所指的地面世界坐标（x,z）。"""
        if self._ui_hovered():
            return None
        wp = mouse.world_point
        if wp is None:
            return None
        return float(wp.x), float(wp.z)

    def _obstacle_root_under_cursor(self) -> Optional[Entity]:
        if self._ui_hovered():
            return None
        e = mouse.hovered_entity
        roots = {ob.entity for ob in self.obstacles}
        while e is not None:
            if e in roots:
                return e
            e = e.parent
        return None

    def _update_ghost(self) -> None:
        if self.mode != MODE_ADD:
            self.ghost.enabled = False
            self.hover_point = None
            return
        hp = self._world_hover()
        if hp is None:
            self.ghost.enabled = False
            self.hover_point = None
            return
        x, z = hp
        k = OB_KINDS[self.add_kind]
        ok = self._can_place(x, z, k['w'], k['d'])
        self.ghost.enabled = True
        self.ghost.position = (x, 0.02, z)
        self.ghost.scale = (k['w'], k['h'], k['d'])
        self.ghost.color = (color.rgba32(60, 220, 130, 90) if ok
                            else color.rgba32(235, 80, 80, 90))
        self.hover_point = (x, z) if ok else None

    def on_click(self) -> None:
        if self._ui_hovered():
            return
        if self.mode == MODE_TARGET:
            # 只有点到空闲地面才重设目标
            if mouse.hovered_entity is self.ground:
                hp = self._world_hover()
                if hp is not None:
                    self.paused = False
                    self.set_goal(Vec2(*hp), count_replan=False)
        elif self.mode == MODE_ADD:
            if self.hover_point is not None:
                self.add_obstacle(self.add_kind, *self.hover_point)
        else:  # MODE_DELETE
            root = self._obstacle_root_under_cursor()
            if root is not None:
                self.remove_obstacle(root)

    # ========================== 每帧仿真 ==========================
    def update(self) -> None:
        if not self._ui_ready:
            self._build_ui()
            self._ui_ready = True
        dt = min(time_dt(), 0.05)
        self._update_ghost()
        self.turret.rotation_y += dt * 240
        if self.goal_marker.enabled:
            self.goal_ring.rotation_y += dt * 90
            self.goal_beacon.y = 2.45 + 0.12 * math.sin(u_time.time() * 4)

        if self.paused:
            self.status = '暂停'
            self._update_hud()
            return
        if self.goal_pos is None:
            self.status = '待机'
            self._update_hud()
            return

        dist_to_goal = math.hypot(self.robot.x - self.goal_pos.x,
                                  self.robot.z - self.goal_pos.y)
        if dist_to_goal < STOP_TOL:
            self.speed = max(0.0, self.speed - ACCEL * 2.5 * dt)
            if self.speed < 0.05:
                self.speed = 0.0
                self.status = '已到达'
            self.robot.rotation_y = self.heading
            self._roll_wheels(dt)
            self._update_hud()
            return

        if not self.path:
            self.speed = max(0.0, self.speed - ACCEL * 2.5 * dt)
            self.status = '无可行路径'
            self._update_hud()
            return

        # ---- 前方路径被新障碍阻断 -> 自动重新规划（带节流） ----
        self.block_cd -= dt
        if self._lookahead_blocked() and self.block_cd <= 0:
            self.block_cd = 0.3
            if self._recompute(count_replan=True):
                self.status = '重新规划'
            else:
                self.status = '无可行路径'
                self._update_hud()
                return

        # ---- 转向（大角度偏差时原地转向，避免带速挤进墙根） ----
        tgt = self.path[0]
        dx, dz = tgt[0] - self.robot.x, tgt[1] - self.robot.z
        target_yaw = math.degrees(math.atan2(dx, dz))
        diff = (target_yaw - self.heading + 180) % 360 - 180
        self.heading += clamp(diff, -MAX_ANGULAR * dt, MAX_ANGULAR * dt)

        # ---- 速度规划 ----
        if abs(diff) > 40.0:
            target_speed = 0.0                      # 原地掉头 / 转向
        else:
            align = clamp(1.0 - (abs(diff) - 8.0) / 32.0, 0.25, 1.0) \
                if abs(diff) > 8.0 else 1.0
            target_speed = MAX_SPEED * align
            remain = self.remaining_distance()
            if remain < 2.2:
                # 末段线性减速，允许趋近于 0，保证精确停在终点
                target_speed = min(target_speed, remain * 1.6 + 0.05)
        if self.speed < target_speed:
            self.speed = min(target_speed, self.speed + ACCEL * dt)
        else:
            self.speed = max(target_speed, self.speed - ACCEL * 2.0 * dt)

        # ---- 位移积分（末级碰撞保险） ----
        yaw_r = math.radians(self.heading)
        step = self.speed * dt
        nx = self.robot.x + math.sin(yaw_r) * step
        nz = self.robot.z + math.cos(yaw_r) * step
        if self.world.is_free_world(nx, nz):
            self.robot.x, self.robot.z = nx, nz
        else:
            self.speed = 0.0
            if self.block_cd <= 0:
                self.block_cd = 0.3
                if not self._recompute(count_replan=True):
                    self.status = '无可行路径'
        self.robot.rotation_y = self.heading
        self._roll_wheels(dt)

        # ---- 消费路径点 ----
        if self.path and math.hypot(self.path[0][0] - self.robot.x,
                                    self.path[0][1] - self.robot.z) < WP_TOL:
            self.path.pop(0)
            self.render_path()

        # ---- 行驶轨迹 ----
        if self.speed > 0.05:
            if not self.trail or math.hypot(self.trail[-1][0] - self.robot.x,
                                            self.trail[-1][1] - self.robot.z) \
                    > 0.22:
                self.add_trail_point((self.robot.x, self.robot.z))
            self.status = '前往目标'
        self._update_hud()

    def _lookahead_blocked(self) -> bool:
        """检测规划路径前方 1.4m 内的线段是否被新障碍阻塞。

        沿路径点依次累加距离，只检测路径实际覆盖到的范围；
        剩余路径不足 1.4m 时就只检测到终点，绝不向终点之外外推
        （终点在围栏边时外推会探到边界安全带而误判阻塞）。
        """
        if not self.path:
            return False
        p0 = (self.robot.x, self.robot.z)
        probe_d = 1.4
        prev = p0
        covered = 0.0
        for pt in self.path:
            seg = math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            if covered + seg >= probe_d:
                t = (probe_d - covered) / seg if seg > 1e-9 else 0.0
                probe = (prev[0] + (pt[0] - prev[0]) * t,
                         prev[1] + (pt[1] - prev[1]) * t)
                return self.world.segment_blocked(prev, probe) \
                    or self.world.segment_blocked(p0, prev)
            if self.world.segment_blocked(prev, pt):
                return True
            covered += seg
            prev = pt
        return False  # 整条剩余路径短于 1.4m 且全部畅通

    def _roll_wheels(self, dt: float) -> None:
        # 车轮半径 0.20，按 v/r 绕车轴（枢轴 X 轴）滚动
        ang = math.degrees(self.speed * dt / 0.20)
        for w_ in self.wheels:
            w_.rotation_x += ang

    # ========================== 重置 ==========================
    def reset_robot(self, keep_goal: bool = True) -> None:
        self.robot.position = (START_POS[0], 0, START_POS[1])
        self.heading = 0.0
        self.robot.rotation = (0, 0, 0)
        self.speed = 0.0
        self.path = []
        self.clear_trail()
        self.clear_path_viz()
        self.replan_count = 0
        self.status = '待机'
        if not keep_goal:
            self.goal_pos = None
            self.goal_marker.enabled = False

    def restart(self) -> None:
        self.paused = False
        self.reset_robot(keep_goal=True)
        if self.goal_pos is not None:
            self._recompute(count_replan=False)

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def set_mode(self, m: str) -> None:
        self.mode = m
        self._refresh_mode_buttons()

    def cycle_add_kind(self) -> None:
        self.add_kind = KIND_ORDER[
            (KIND_ORDER.index(self.add_kind) + 1) % len(KIND_ORDER)]
        self.kind_text.text = f'添加类型：{self.add_kind}（Q 切换）'

    # ========================== 状态灯 / HUD ==========================
    def _set_status_light(self) -> None:
        table = {
            '前往目标': color.rgb32(60, 220, 130),
            '已到达': color.rgb32(80, 235, 140),
            '重新规划': color.rgb32(255, 190, 60),
            '无可行路径': color.rgb32(235, 70, 70),
            '暂停': color.rgb32(250, 220, 90),
            '待机': color.cyan,
        }
        c = table.get(self.status, color.cyan)
        if self.status in ('无可行路径', '重新规划') \
                and int(u_time.time() * 4) % 2 == 0:
            c = color.white
        self.status_light.color = c

    def _build_ui(self) -> None:
        # 所有 UI 挂在一个位于原点的容器下：window 的宽高比自动修正只遍历
        # parent 为 camera.ui 的直接子节点，容器 x=0 修正后不变，
        # 内部元素坐标因此保持稳定。
        ui_root = Entity(parent=camera.ui)
        ui_root._is_ui = True
        vx = 0.8  # ui 本地坐标水平可视半宽（世界半宽/ui缩放）

        def mark(e: Entity) -> Entity:
            e._is_ui = True
            e.parent = ui_root
            return e

        panel_w = 0.275
        panel_x = -vx + 0.02 + panel_w / 2
        Entity(parent=ui_root, model='quad',
               scale=(panel_w, 0.335), position=(panel_x, 0.285),
               color=color.rgba32(28, 32, 40, 215))._is_ui = True
        self.hud = Text(parent=ui_root,
                        position=(-vx + 0.035, 0.43),
                        scale=0.92, line_height=1.18,
                        color=color.rgb32(235, 240, 248))
        self.hud._is_ui = True
        self.kind_text = Text(parent=ui_root, position=(0.0, -0.392),
                              scale=0.8, color=color.rgb32(220, 226, 236),
                              origin=(0, 0))
        self.kind_text._is_ui = True
        t1 = Text(parent=ui_root, position=(0, 0.472), origin=(0, 0),
                  scale=1.05, text='智能仓储 AGV 路径规划仿真',
                  color=color.rgb32(40, 50, 66))
        t1._is_ui = True
        t2 = Text(parent=ui_root, position=(0, 0.425), origin=(0, 0),
                  scale=0.7,
                  text='左键：当前模式操作    Tab/1/2/3：切换模式    Q：障碍类型\n'
                       '空格：暂停/继续    R：重新开始    N：随机场景',
                  color=color.rgb32(90, 100, 116))
        t2._is_ui = True

        # 底部按钮：在可视区内均匀排布
        bw, bh, gap = 0.155, 0.065, 0.012
        labels = [
            ('目标点 (1)', MODE_TARGET, color.rgb32(70, 80, 96), 'mode'),
            ('添加障碍 (2)', MODE_ADD, color.rgb32(70, 80, 96), 'mode'),
            ('删除障碍 (3)', MODE_DELETE, color.rgb32(70, 80, 96), 'mode'),
            ('暂停/继续', None, color.rgb32(58, 92, 84), 'act_pause'),
            ('重新开始 R', None, color.rgb32(58, 92, 84), 'act_restart'),
            ('随机场景 N', None, color.rgb32(58, 92, 84), 'act_random'),
        ]
        total = len(labels) * bw + (len(labels) - 1) * gap
        x0 = -total / 2 + bw / 2
        self.mode_buttons = {}
        for i, (label, key, col, kind) in enumerate(labels):
            bx = x0 + i * (bw + gap)
            b = Button(parent=ui_root, text=label,
                       scale=(bw, bh), position=(bx, -0.455),
                       color=col,
                       highlight_color=color.rgb32(110, 126, 146),
                       text_size=0.78)
            b._is_ui = True
            b.text_entity.color = color.white
            b.text_entity._is_ui = True
            if kind == 'mode':
                self.mode_buttons[key] = b
            elif kind == 'act_pause':
                b.on_click = self.toggle_pause
            elif kind == 'act_restart':
                b.on_click = self.restart
            elif kind == 'act_random':
                b.on_click = self.random_scene
        self._refresh_mode_buttons()

    def _refresh_mode_buttons(self) -> None:
        for name, b in self.mode_buttons.items():
            if name == self.mode:
                b.color = color.rgb32(230, 150, 40)
                b.highlight_color = color.rgb32(240, 170, 60)
            else:
                b.color = color.rgb32(70, 80, 96)
                b.highlight_color = color.rgb32(96, 110, 130)
        if self.kind_text is not None:
            self.kind_text.text = (f'添加类型：{self.add_kind}（Q 切换）'
                                   if self.mode == MODE_ADD else '')

    def _update_hud(self) -> None:
        self._set_status_light()
        if self.hud is None:
            return
        self.hud.text = (
            f'状态：{self.status}\n'
            f'速度：{self.speed:5.2f} m/s\n'
            f'路径长度：{path_length(self.path):7.2f} m\n'
            f'剩余距离：{self.remaining_distance():7.2f} m\n'
            f'重新规划次数：{self.replan_count}\n'
            f'当前模式：{self.mode}\n'
            f'障碍数量：{len(self.obstacles)}'
        )

    def input(self, key: str) -> None:
        if key == 'left mouse down':
            self.on_click()
        elif key == 'tab':
            order = [MODE_TARGET, MODE_ADD, MODE_DELETE]
            self.set_mode(order[(order.index(self.mode) + 1) % 3])
        elif key == '1':
            self.set_mode(MODE_TARGET)
        elif key == '2':
            self.set_mode(MODE_ADD)
        elif key == '3':
            self.set_mode(MODE_DELETE)
        elif key == 'q':
            self.cycle_add_kind()
        elif key == 'space':
            self.toggle_pause()
        elif key == 'r':
            self.restart()
        elif key == 'n':
            self.random_scene()
        elif key == 'escape':
            application.quit()


def time_dt() -> float:
    return float(u_time.dt)

# ========================== 入口 ==========================
if __name__ == '__main__':
    sim: Optional[AGVSimulation] = None

    def update() -> None:
        sim.update()

    def input(key: str) -> None:
        sim.input(key)

    sim = AGVSimulation()

    if os.environ.get('AGV_SMOKE'):
        # 无窗口自测：步进若干帧，在规划路径前方放置挡路障碍，
        # 验证真实运动、碰撞拦截与自动重规划
        task_mgr = sim.app.taskMgr
        start_pos = (sim.robot.x, sim.robot.z)
        blocked_at = None
        frames = int(os.environ.get('AGV_SMOKE', '600'))
        for i in range(frames):
            task_mgr.step()
            if i == 120 and sim.status == '前往目标' and len(sim.path) >= 2:
                # 沿当前规划路径前方 ~6m 处取点，放置一个合法且能挡住路径的箱子
                cum = 0.0
                bp = sim.path[0]
                for a, b in zip(sim.path, sim.path[1:]):
                    seg = math.hypot(b[0] - a[0], b[1] - a[1])
                    if cum + seg >= 6.0:
                        t = (6.0 - cum) / seg
                        bp = (a[0] + (b[0] - a[0]) * t,
                              a[1] + (b[1] - a[1]) * t)
                        break
                    cum += seg
                    bp = b
                # 设备箱较窄，是 _can_place 允许且足以阻断单条路径的障碍
                if sim.add_obstacle('设备箱', bp[0], bp[1]):
                    blocked_at = bp
        moved = math.hypot(sim.robot.x - start_pos[0],
                           sim.robot.z - start_pos[1])
        print(f'[smoke] moved={moved:.2f}m replans={sim.replan_count} '
              f'status={sim.status} trail_pts={len(sim.trail)} '
              f'pos=({sim.robot.x:.1f},{sim.robot.z:.1f}) '
              f'block_at={blocked_at and tuple(round(v,1) for v in blocked_at)}')
        assert blocked_at is not None, '未能在路径上放置障碍'
        assert moved > 5.0, '机器人没有真实移动'
        assert sim.replan_count >= 1, '阻断后未触发重新规划'
        assert sim.status in ('前往目标', '已到达'), '最终状态异常'
        print('[smoke] PASS')
        sys.exit(0)

    sim.app.run()
