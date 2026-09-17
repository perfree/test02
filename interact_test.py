"""在机器人前方路径上动态放置障碍，验证自动重规划与绕行。"""
from panda3d.core import loadPrcFileData
loadPrcFileData('', 'sync-video 0')
loadPrcFileData('', 'clock-mode limited')
loadPrcFileData('', 'clock-frame-rate 60')
loadPrcFileData('', 'win-size 1400 900')
loadPrcFileData('', 'fullscreen 0')

import math
from ursina import *
import agv_sim as M

app = Ursina(development_mode=False, fullscreen=False)
window.borderless = False
window.color = color.hsv(212, .22, .92)
window.fps_counter.enabled = False
window.exit_button.visible = False

sim = M.AGVSimulation()
sim._app = app

class Driver(Entity):
    def __init__(self):
        super().__init__()
        self.n = 0
        self.placed = False
        self.rc_at_place = 0
        self.rc_before_total = 0
    def update(self):
        self.n += 1
        if self.n == 150 and not self.placed:
            # 沿路径走约 5m 取前方走廊点放集装箱，切断当前路径
            need = 5.0
            wp = sim.path[-1]
            for j in range(len(sim.path) - 1):
                seg = math.hypot(sim.path[j+1].x-sim.path[j].x,
                                 sim.path[j+1].z-sim.path[j].z)
                if seg >= need:
                    t = need / seg
                    wp = M.Vec3(sim.path[j].x + (sim.path[j+1].x-sim.path[j].x)*t,
                                0,
                                sim.path[j].z + (sim.path[j+1].z-sim.path[j].z)*t)
                    break
                need -= seg
            cell = sim.planner.world_to_cell(wp)
            rc_before = sim.replan_count
            ok = sim.add_obstacle_at('container', cell, 0, quiet=True)
            if not ok:
                ok = sim.add_obstacle_at('equip', cell, 0, quiet=True)
            self.placed = ok
            self.rc_before_total = sim.replan_count
            new = [sim.planner.world_to_cell(p) for p in sim.path]
            blocked_cells = {(cell[0]+dx, cell[1]+dz)
                             for dx in (-1,0,1) for dz in (-1,0,1)}
            crosses = any(c in blocked_cells for c in new)
            print('PLACE ok=', ok, 'cell=', cell,
                  'replan+', sim.replan_count - rc_before,
                  'path_crosses_blocked=', crosses, flush=True)
            assert ok, '障碍必须放置成功'
            assert sim.replan_count > rc_before, '放置障碍后应触发重规划'
            assert not crosses, '新路径必须绕开障碍膨胀区'
        if self.n == 420:
            print('AFTER state=', sim.robot_state, 'pathlen=', len(sim.path), flush=True)
            base.screenshot('interact_shot.png', defaultFilename=False)
            assert len(sim.path) >= 1, '必须存在路径'
            assert sim.replan_count - self.rc_before_total <= 3, (
                f'不应出现重规划风暴: +{sim.replan_count - self.rc_before_total}')
            print('INTERACT_OK total_replan=', sim.replan_count, flush=True)
            application.quit()

Driver()
app.run()
