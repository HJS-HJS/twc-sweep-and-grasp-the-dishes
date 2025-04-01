import sys
import os
import time
import yaml
import numpy as np
import random
import copy
from typing import List

base_dir = os.path.dirname(os.path.abspath(__file__))
so_file_path = os.path.join(base_dir, "../../cpp")
sys.path.append(os.path.abspath(so_file_path))

from quasi_static_push import SimulationViewer, SimulationResult, SimulationDoneReason, GripperMotion, Player

class Simulation():
    def __init__(self, visualize:str = 'human', state:str = 'image', action_skip:int = 5, record:bool = False, save_dir:str = "recordings"):
        """
        state : image, information
        """
        # Get initial param
        self.state = state

        # Set display visuality
        if visualize == "human": visualize = True
        elif visualize is None: visualize = False
        else: visualize = False

        ## Get config file
        with open(os.path.dirname(os.path.abspath(__file__)) + "/../../config/simulation.yaml") as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)

        # Set patameters
        self.display_size = np.array([self.config["display"]["WIDTH"], self.config["display"]["HEIGHT"] ]) # Get display size parameter from config.yaml

        ## Set parameters
        # Set pixel unit
        self.unit = self.config["display"]["unit"] #[m/pixel]

        # Set pusher
        _pusher_type = self.config["pusher"]["pusher_type"]

        # Set pusher unit speed
        unit_v_speed = self.config["pusher"]["unit_v_speed"] # [m/s]
        unit_r_speed = self.config["pusher"]["unit_r_speed"] # [rad/s]
        unit_w_speed = self.config["pusher"]["unit_w_speed"] # [m/s]
        self.unit_speed = [unit_v_speed, unit_v_speed, unit_r_speed, unit_w_speed, int(1)]

        self.pusher_width_limit = np.array([self.config["pusher"]["pusher_d_u_limit"], self.config["pusher"]["pusher_d_l_limit"]])

        # Set slider 
        self.slider_max_num = self.config["auto"]["maximun_number"] # Get sliders number
        self.min_r = self.config["auto"]["minimum_radius"]
        self.max_r = self.config["auto"]["maximum_radius"]

        # Set simulation param
        self.fps = self.config["simulator"]["fps"]
        self.action_skip = action_skip

        self.pusher_input = [
            self.config["pusher"]["pusher_num"], self.config["pusher"]["pusher_angle"], _pusher_type["type"], 
            {"a": _pusher_type["a"], "b": _pusher_type["b"], "n": _pusher_type["n"]}, 
            self.config["pusher"]["pusher_distance"], self.pusher_width_limit[0], self.pusher_width_limit[1],
            0.0, 0.0, 0.0
        ]

        self.action_limit = np.array([
            [-1., 1.],
            [-1., 1.],
            [-1., 1.],
            [-1., 1.],
            [0, 1],
        ])

        self.action_space = np.zeros_like(self.unit_speed)
        if state == "image" :   self.observation_space = np.zeros((self.config["display"]["WIDTH"],self.config["display"]["HEIGHT"],3))
        elif state == "gray":   self.observation_space = np.zeros((self.config["display"]["WIDTH"],self.config["display"]["HEIGHT"],1))
        elif state == "linear": self.observation_space = np.zeros(1 + 2 + 4 + 5), np.zeros((2, 5))

        self.simulator = SimulationViewer(
            window_width = self.display_size[0],
            window_height = self.display_size[1],
            scale = 1 / self.unit,
            headless = not visualize,
            gripper_movement = GripperMotion.MOVE_TO_TARGET,
            # gripper_movement = GripperMotion.MOVE_XY,
            # gripper_movement = GripperMotion.MOVE_FORWARD,
            frame_rate = self.fps,
            frame_skip = self.action_skip,
            # grid = False,
            grid = True,
            recording_enabled = record,
            recording_path = save_dir,
            show_closest_point = False,
            )
        
    def reset(self, 
              table_size:List[float] = None,
              slider_state:List[List[float]] = None,
              slider_num:int=None,
              ):
                        
            # Table setting
            if table_size is None:
                _table_limit_width  = random.randint(int(self.display_size[0] * 0.3), int(self.display_size[0] * 0.6))
                _table_limit_height = random.randint(int(self.display_size[1] * 0.3), int(self.display_size[1] * 0.6))
                _table_limit = np.array([_table_limit_width, _table_limit_height])
                table_size = _table_limit * self.unit
            else:
                _table_limit = (np.array(table_size) / self.unit).astype(int)
            self.table_limit = _table_limit * self.unit / 2

            # Slider setting
            if slider_state is None:
                slider_inputs = []
                _slider_num = random.randint(1, self.slider_max_num) if slider_num is None else np.clip(slider_num, 1, 15)
                points, radius = self.generate_spawn_points(_slider_num)
                for point, _r in zip(points, radius):
                    a = np.clip(random.uniform(0.8, 1.0) * _r, a_min=self.min_r, a_max=_r)
                    b = np.clip(random.uniform(0.75, 1.25) * a, a_min=self.min_r, a_max=_r)
                    r = random.uniform(0, np.pi * 2)
                    slider_inputs.append(("ellipse", [point[0], point[1], r, a, b]))
            else:
                slider_inputs = []
                # print(slider_inputs)
                for slider in slider_state:
                    # print(slider)
                    if len(slider) == 4: slider_inputs.append(("circle", [slider[0], slider[1], slider[2], slider[3]]))
                    else               : slider_inputs.append(("ellipse",[slider[0], slider[1], slider[2], slider[3], slider[4]]))
            slider_num = len(slider_inputs)
            
            # Initial pusher pose
            self.pusher_input[4] = random.uniform(self.pusher_input[5], self.pusher_input[6])    # width
            self.pusher_input[7] = 0   # x
            self.pusher_input[8] = 0   # y
            self.pusher_input[9] = 0   # w

            self.state_prev = None

            self.simulator.reset(
                slider_inputs = slider_inputs,
                pusher_input = tuple(self.pusher_input),
                newtableWidth = self.table_limit[0] * 2,
                newtableHeight = self.table_limit[1] * 2,
            )

            return self.step([0.])

    def image_without_gripper(self):
        self.simulator.renderViewer_(False)
        return self.simulator.getImageState()

    def step(self, action, mode:int = 1):
        if(len(action) == 3): action = np.hstack((action, 1))

        if len(action) == 1:
            state_curr = self.simulator.run([0., 0., 0. ,0., 0.])
            mode = 0
        elif mode == 0:
            if(len(action) != 4): print("Invalid position size")
            action = np.clip(action, self.action_limit[:4, 0], self.action_limit[:4, 1])
            action[:2] *= self.display_size / 2 * self.unit
            action[2] *= np.pi / 3
            action[3] = self.pusher_width_limit[1] + (action[3] / 2 + 0.5) * (self.pusher_width_limit[0] - self.pusher_width_limit[1])
            self.simulator.applyGripperPosition(action)

            state_curr = self.simulator.run([0., 0., 0., 0., 1])

        elif mode == 1:
            if(len(action) != 4): print("Invalid action space")
            
            action = np.clip(action, self.action_limit[:4, 0], self.action_limit[:4, 1])
            action *= self.unit_speed[:4]
            
            state_curr = self.simulator.run(np.hstack((action, 1)))

        time.sleep(0.01)

        result = self.get_results(self.state_prev, state_curr, mode)
        self.state_prev = state_curr

        return result
    
    def get_results(self, state_prev: SimulationResult = None, state_curr: SimulationResult = None, mode:int = 0):
        # State
        # width = self.display_size * self.unit / 2
        _table = ((self.table_limit - 0.25) / 0.25) * 2 - 1
        _pusher = copy.deepcopy(state_curr.pusher_state)
        if len(state_curr.slider_state) == 0:
            _sliders = copy.deepcopy(np.array(state_prev.slider_state))
        else:
            _sliders = copy.deepcopy(np.array(state_curr.slider_state))

        _pusher_width = copy.deepcopy(_pusher[3])
        _pusher[3] = (_pusher[3] - self.pusher_width_limit[1]) / (self.pusher_width_limit[0] - self.pusher_width_limit[1]) * 2 - 1

        _sliders_theta_sin = np.sin(_sliders[:,2])
        _sliders_theta_cos = np.cos(_sliders[:,2])
        _sliders[:,3:5] = (_sliders[:,3:5] - self.min_r) / (self.max_r - self.min_r) * 2 - 1

        fingers = np.array([
            [_pusher[0] +  np.cos(_pusher[2] + np.pi / 6) * _pusher_width,
             _pusher[1] +  np.sin(_pusher[2] + np.pi / 6) * _pusher_width],
            [_pusher[0] +  np.cos(_pusher[2] + np.pi / 6 + np.pi * 2 / 3) * _pusher_width,
             _pusher[1] +  np.sin(_pusher[2] + np.pi / 6 + np.pi * 2 / 3) * _pusher_width],
            [_pusher[0] +  np.cos(_pusher[2] + np.pi / 6 + np.pi * 4 / 3) * _pusher_width,
             _pusher[1] +  np.sin(_pusher[2] + np.pi / 6 + np.pi * 4 / 3) * _pusher_width],
        ])

        if mode <= 0:
            _pusher = np.zeros_like(_pusher)
            fingers = np.zeros_like(fingers)
        _state1 = np.hstack((
                _table,
                _sliders[0][:2] * 2,
                [(_sliders[0][0] +  _sliders[0][3] * _sliders_theta_cos[0]) * 2,
                 (_sliders[0][1] +  _sliders[0][3] * _sliders_theta_sin[0]) * 2,
                 (_sliders[0][0] + -_sliders[0][4] * _sliders_theta_sin[0]) * 2,
                 (_sliders[0][1] +  _sliders[0][4] * _sliders_theta_cos[0]) * 2,
                 (_sliders[0][0] + -_sliders[0][3] * _sliders_theta_cos[0]) * 2,
                 (_sliders[0][1] + -_sliders[0][3] * _sliders_theta_sin[0]) * 2,
                 (_sliders[0][0] +  _sliders[0][4] * _sliders_theta_sin[0]) * 2,
                 (_sliders[0][1] + -_sliders[0][4] * _sliders_theta_cos[0]) * 2,
                ],
                (len(_sliders) / 5) - 1,
                _pusher[:2] * 2,
                fingers.reshape(-1) * 2,
                ))

        _state2 = np.zeros(((len(_sliders)), 19))

        for idx in range(0, len(_sliders)):
            # if mode <= 0:
            #     relative_pusher_target = np.zeros_like(fingers)
            # else:
            #     relative_pusher_target = _sliders[idx][:2] - fingers
            
            # relative_slider_target = (_sliders[idx][:2] - _sliders[0][:2])

            if idx == 0: _target = -1
            else:        _target = 1

            _state2[idx] = np.concatenate([
                _table,
                [_target],
                _sliders[idx][:2] * 2,
                [(_sliders[idx][0] +  _sliders[idx][3] * _sliders_theta_cos[idx]) * 2,
                 (_sliders[idx][1] +  _sliders[idx][3] * _sliders_theta_sin[idx]) * 2,
                 (_sliders[idx][0] + -_sliders[idx][4] * _sliders_theta_sin[idx]) * 2,
                 (_sliders[idx][1] +  _sliders[idx][4] * _sliders_theta_cos[idx]) * 2,
                 (_sliders[idx][0] + -_sliders[idx][3] * _sliders_theta_cos[idx]) * 2,
                 (_sliders[idx][1] + -_sliders[idx][3] * _sliders_theta_sin[idx]) * 2,
                 (_sliders[idx][0] +  _sliders[idx][4] * _sliders_theta_sin[idx]) * 2,
                 (_sliders[idx][1] + -_sliders[idx][4] * _sliders_theta_cos[idx]) * 2,
                ],
                fingers.reshape(-1) * 2,
            ])

        state = _state1, _state2

        if self.state == "linear":
            pass
        elif self.state == "image":
            state = state_curr.image_state, _state1, _state2
        else:                        state = None

        # Reward
        if mode == 0: reward = self.cal_init_reward(state_prev, state_curr)
        else:         reward = self.cal_reward(state_prev, state_curr)

        # Mode
        if state_curr.mode < 0:      mode = 0
        else:                        mode = 1

        return state, reward, state_curr.done, mode 

    def cal_init_reward(self, state_prev: SimulationResult, state_curr: SimulationResult):
        ## Reward
        reward = 0.0

        if state_prev is None: return 0
        if state_curr.done & SimulationDoneReason.DONE_GRASP_SUCCESS.value:
            print("DONE_GRASP_SUCCESS")
            reward += 15.0
        if state_curr.done & SimulationDoneReason.DONE_GRASP_FAILED.value:
            print("DONE_GRASP_FAILED")
            return -20.0

        # Spawn failed penalty
        if state_prev.mode == state_curr.mode:
            return -1.0
        else:
            reward += 1.0

        ## Pusher distance from target
        if len(state_curr.slider_state) == 0:
            pusher_distance = (state_curr.pusher_state[:2] - np.array([0, 0]))
        else:
            pusher_distance = (state_curr.pusher_state[:2] - state_curr.slider_state[0][:2])
        pusher_distance = np.linalg.norm(pusher_distance)

        reward += 6.0 * (1 - pusher_distance / 0.2)

        return reward
        
    def augment_init_data(self, state1 = np.array([0, 0])):
        state = copy.deepcopy(state1)
        width = self.display_size * self.unit / 2

        action = np.zeros(4)

        _pusher = self.state_prev.pusher_state

        action[:2] = _pusher[:2] / (self.display_size * self.unit / 2)
        action[2] = _pusher[2] / np.pi
        action[3] = (_pusher[3] - self.pusher_width_limit[1]) / (self.pusher_width_limit[0] - self.pusher_width_limit[1]) * 2 - 1

        ## Pusher distance from target
        pusher = state[2:4]
        target = state[7:9]
        pusher_distance = (pusher - target) * width
        pusher_distance = np.linalg.norm(pusher_distance)
    
        reward = 0
        if pusher_distance < 0.005:
            reward += 20.0
        reward += 10.0
        reward += 12.0 * (1 - pusher_distance / 0.2)

        state[2:7] = np.zeros_like(state[2:7])
        
        return state, reward, action.astype(np.float32)

    def cal_reward(self, state_prev: SimulationResult, state_curr: SimulationResult):
        if state_prev is None: return 0

        ## Reward
        reward = -1.5

        ## Failed
        if state_curr.done & SimulationDoneReason.DONE_FALL_OUT.value:
            print("DONE_FALL_OUT")
            return -20.0
        if state_curr.done & SimulationDoneReason.DONE_GRASP_SUCCESS.value:
            print("DONE_GRASP_SUCCESS")
            return 15.0
        if state_curr.done & SimulationDoneReason.DONE_GRASP_FAILED.value:
            print("DONE_GRASP_FAILED")
            return -20.0

        ## Pusher distance from target
        pusher_distance_prev = np.linalg.norm(state_prev.pusher_state[:2] - state_prev.slider_state[0][:2])
        if len(state_curr.slider_state) == 0:
            pusher_distance_curr = pusher_distance_prev
        else:
            pusher_distance_curr = np.linalg.norm(state_curr.pusher_state[:2] - state_curr.slider_state[0][:2])
        pusher_distance_diff = (pusher_distance_prev - pusher_distance_curr) * self.fps / self.action_skip

        # velocity
        reward += 15.0 * pusher_distance_diff
        # distance
        reward += 1.3 * (0.4 - pusher_distance_curr) / 0.4

        ## Slider
        if len(state_prev.slider_state) != len(state_curr.slider_state):
            reward += -1.0
        else:
            slider_distance = np.linalg.norm((np.array(state_prev.slider_state)[:,:2] - np.array(state_curr.slider_state)[:,:2]), axis=1)
            slider_distance_diff = slider_distance * self.fps / self.action_skip

            _delta_slider_dist = np.where(np.abs(slider_distance_diff[1:]) - 1e-5 > 0)[0]
            if len(_delta_slider_dist) > 0:
                reward += -1.0
            if slider_distance_diff[0] - 1e-5 > 0:
                reward += -0.8
            
            # Simulation break case
            if np.max(np.abs(slider_distance_diff)) > 0.2:
                print("SIMULATION BREAK")
                reward = -1000

            # Check danger
            slider_distance = (
                (np.abs(np.array(state_prev.slider_state)[:,:2]) - np.abs(np.array(state_curr.slider_state)[:,:2]))
                ).reshape(-1)
            danger_list = np.array(state_curr.slider_state)[:,:2]

            danger_list1 = ((np.abs(danger_list) + self.min_r * 2.0 - self.table_limit) / (self.min_r * 2.0)).reshape(-1)
            danger_list1[np.where(danger_list1 < 0.0)[0]] = 0

            danger_list1[np.where(np.abs(slider_distance) - 1e-9 < 0)[0]] = 0
            danger_list1[np.where(slider_distance - 1e-9 > 0)[0]] *= -1

            danger_list_num1 = danger_list1
            reward += -10.0 * (np.sum(danger_list_num1) / 3)

        return reward
    
    def get_state(self):
        table_size = self.table_limit * 2
        slider_state = self.state_prev.slider_state
        return {"table_size":table_size, "slider_state":slider_state}

    def generate_spawn_points(self, num_points, center_bias=0.7):
        points = []
        x_range = ((-self.table_limit[0] + self.min_r * 1.3) * 0.9, (self.table_limit[0] - self.min_r * 1.3) * 0.9)
        y_range = ((-self.table_limit[1] + self.min_r * 1.3) * 0.9, (self.table_limit[1] - self.min_r * 1.3) * 0.9)

        # 첫 번째 점을 랜덤하게 생성
        center_x = random.uniform(*x_range)
        center_y = random.uniform(*y_range)
        points.append((center_x, center_y))

        # Raduis of inital point
        init_r = random.uniform(self.min_r, self.max_r)
        available_lengh = (init_r + self.min_r, init_r + self.max_r)
        
        # 나머지 점 생성
        candidate_points = []
        for _ in range(num_points - 1):
            # 첫 번째 점 주변에서 가우시안 분포로 점 생성
            if random.random() < center_bias:  # 중심 근처에 생성될 확률
                new_x = np.clip(np.random.normal(center_x, random.uniform(*available_lengh)), *x_range)
                new_y = np.clip(np.random.normal(center_y, random.uniform(*available_lengh)), *y_range)
            else:  # 전체 영역에 균일 분포로 생성
                new_x = random.uniform(*x_range)
                new_y = random.uniform(*y_range)
            candidate_points.append((new_x, new_y))
        
        # 거리 조건을 만족하는 점만 선택
        for point in candidate_points:
            distances = [np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) for p in points]
            if all(d >= (init_r + self.min_r) for d in distances):
                points.append(point)
        
        points = np.array(points)

        min_distances = np.ones(len(points)) * self.min_r
        min_distances[0] = init_r

        for idx, point in enumerate(points):
            if idx == 0: continue
            distances = [np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) for p in points]
            distances = distances - min_distances
            distances[idx] = self.max_r
            min_distances[idx] = min(distances)

        # 첫 번째 점을 포함한 최종 점 리스트
        return points, np.array(min_distances)
    def get_image(self):
        return self.state_prev.image_state

class DishSimulation():
    def __init__(self, visualize:str = 'human', state:str = 'image', action_skip:int = 5, record:bool = False, save_dir:str = "recordings"):
        self.env = Simulation(visualize = visualize, state = state, action_skip = action_skip, record = record, save_dir=save_dir)
        self._count = 0
        self._setting = None
        self.save_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"
        self.skip_frame = 0
    
    def reset(self, mode:str=None, slider_num:int = 15):
        if mode == None: 
            state_curr = self.env.reset(slider_num=slider_num)
            self._setting = self.env.get_state()

        elif mode == "continous":
            _setting = self.env.get_state()
            if (self._setting == None) or (len(_setting["slider_state"]) == 0):
                state_curr = self.env.reset(slider_num=slider_num)
                self._setting = _setting
            else:
                state_curr = self.env.reset(
                    table_size  = _setting["table_size"],
                    slider_state = _setting["slider_state"],
                    )
        
        elif mode == "pusher":
            if self._count == 0:
                state_curr = self.env.reset(slider_num=slider_num)
                self._setting = self.env.get_state()
                self._count += 1
            else:
                state_curr = self.env.reset(
                    table_size  = self._setting["table_size"],
                    slider_state = self._setting["slider_state"],
                    )
                self._count = (self._count + 1) % 4

        elif mode == "test":
            _setting = self.env.get_state()
            if (self._setting == None) or (_setting["slider_num"] == 0):
                state_curr = self.env.reset(slider_num=slider_num)
                self._setting = _setting
            else:
                state_curr = self.env.reset(
                    table_size  = _setting["table_size"],
                    slider_state = _setting["slider_state"],
                    )
        else: 
            state_curr = self.env.reset(slider_num=slider_num)
            self._setting = self.env.get_state()
        return state_curr

    def keyboard_control(self):
        return self.env.simulator.keyboard_input()

    def replay_video(self):
        import cv2
        player = Player("recordings", GripperMotion.MOVE_XY)
        state_prev = None
        for is_new, state, action in player:
            if is_new: 
                state_prev = state
                continue

            # cv2.imshow("Replay", state.image_state)
            # time.sleep(1/30)
            # if cv2.waitKey(30) == 27:  # ESC 키로 종료
            #     break
            if state.mode < 0: continue
            yield action[:4], state_prev, self.env.get_results(state_prev, state, state.mode + 1)
            state_prev = state

        cv2.destroyAllWindows()
        del player
        return

if __name__=="__main__":
    # sim = DishSimulation(state='linear', action_skip=8, record=True)
    sim = DishSimulation(state='linear', action_skip=8)
    print("start")

    # for action, (state_next, reward, done, mode) in sim.replay_video():
    #     print(state_next)
    #     pass
    # exit()

    import time
    state = sim.reset(mode=None, slider_num=2)
    state_curr, _, _, mode = sim.reset(mode="continous", slider_num=2)
    state_next, reward, done, mode_next = sim.env.step([random.choice([0.9, -0.9]), random.choice([0.9, -0.9]), 0, 0.5], mode)
    # state = sim.reset(mode=None, slider_num=8)
    action_space = sim.env.action_space.shape[0]
    action = np.zeros(action_space) # Initialize pusher's speed set as zeros 
    step = 0
    while True:
        action, escape, reset = sim.keyboard_control()
        if any(action) != 0.:
            # if action[4]>0.5:
            #     action[4] = np.random.random() * 0.5 + 0.5
            # else:
            #     action[4] = np.random.random() * 0.5
            # action[:4] *= np.random.random(4)
            # action[4] = np.random.random() * 0.5
        
            step += 1
            if step % 10 == 0:
                state_next1, state_next2 = state_next
                _, reward, action = sim.env.augment_init_data(state_next1)
                state_next, reward, done, mode = sim.env.step(action=action, mode=0)
            else:
                state_next, reward, done, mode = sim.env.step(action=action[:4])
                
            state = state_next
            time.sleep(0.01)
            if reset or done:
                state_curr, _, _, mode = sim.reset(mode="continous")
                state_next, reward, done, mode_next = sim.env.step([random.choice([0.9, -0.9]), random.choice([0.9, -0.9]), 0, 0.5], mode)
                step = 0
                time.sleep(1)
            if escape:
                exit()
        else:
            time.sleep(0.01)
            if reset:
                state_curr, _, _, mode = sim.reset(mode="continous")
                state_next, reward, done, mode_next = sim.env.step([random.choice([0.9, -0.9]), random.choice([0.9, -0.9]), 0, 0.5], mode)
                step = 0
            if escape:
                exit()