import gym
from gym import error, spaces, utils
from gym.utils import seeding

from pyrep import PyRep
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import PrimitiveShape
from pyrep.const import TextureMappingMode
from pyrep.objects.dummy import Dummy
from itertools import product, repeat
import time
import imageio

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
from os.path import dirname, join, abspath
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

from mazelib.generate.DungeonRooms import DungeonRooms
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.solve.BacktrackingSolver import BacktrackingSolver

class LabyrinthCtrlDiscreteEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, headless=True, historical_samples = 1, historical_axis=1, level = 0, verbose = 1):
        self.action_space = gym.spaces.discrete.Discrete(9)
        self.difficulty_level = level
        self.verbose = verbose
        states = [-1, 0, +1] # -ve, Neutral, +ve
        self.d_angle = np.deg2rad(1.0) # 

        np.random.seed(42)
        self.lookup = [p for p in product(states, repeat=2)]
        np.random.shuffle(self.lookup)

        self.sample = historical_samples
        self.axis = historical_axis

        self.done = False
        self.np_random, _ = gym.utils.seeding.np_random()
        self.m = None
# todo
# Configuration
#   Randomize background color
#   Randomize maze
#   Randomize only ball start + end
#   Randomize only ball start
#   fixed start & end
        
        self.headless = headless
        self.scene = None
        self.camera_object = None
        self.roll_joint_object = None
        self.pitch_joint_object = None
        self.img = None
        self.rec_video = []
        self.endDummy = None
        self.startDummy = None
        self.wall = None
        self.previous_wall = None
        self.init_time = None
        self.prev_dist_to_goal = 0
        self.local_stuck = 0
        self.last_reward = 0
        self.log_location = 'vid/'+time.ctime()+'/'
        os.makedirs(self.log_location, exist_ok=True)
        self.obs = None
        self.scene_parameter = {
                                'scene-location': 'labyrinthGame.ttt',
                                'scene-input': ['roll_joint', 'pitch_joint'],
                                'scene-input-min-max':[np.deg2rad(-30), np.deg2rad(30)],
                                'scene-output': 'Vision_sensor',
                                'scene-maze': 'wall',
                                'scene-ball': 8.0000e-03
                                }
        self.eps = 1.000e-3
        self.pr = PyRep()
        SCENE_FILE = join(dirname(abspath(__file__)), self.scene_parameter['scene-location'])
        # Launch the application with a scene file in headless mode
        self.pr.launch(SCENE_FILE, headless=self.headless) 
        self.reset_count = 0
        self.reset()
        self.observation_space = gym.spaces.box.Box(low=0, high = 255, shape=self.obs.shape, dtype=np.uint8)
        
    def step(self, action):
        reward = 0#self.last_reward
	    # check for nan and replace with zeros
        # action = np.deg2rad(np.array([x if(not np.isnan(x)) else 0.0 for x in action]))
        # Take action
        act = np.array(self.lookup[action])*self.d_angle # extract angle from look up based act index
        roll_angle = self.roll_joint_object.get_joint_position() # return radian
        pitch_angle = self.pitch_joint_object.get_joint_position() # return radian
        
        target_roll = np.clip((roll_angle + act[0]), self.scene_parameter['scene-input-min-max'][0], self.scene_parameter['scene-input-min-max'][1])
        target_pitch = np.clip((pitch_angle + act[1]), self.scene_parameter['scene-input-min-max'][0], self.scene_parameter['scene-input-min-max'][1])

        self.roll_joint_object.set_joint_target_position(target_roll)
        
        self.pitch_joint_object.set_joint_target_position(target_pitch)
        # step
        # get observation
        # get reward
        
        # target = np.linalg.norm([target_roll, target_pitch])
        # current = np.linalg.norm([roll_angle, pitch_angle])
        self.pr.step()
        # trying = 0
        # while trying<5:#
        #     self.pr.step()
        #     roll_angle = self.roll_joint_object.get_joint_position()
        #     pitch_angle = self.pitch_joint_object.get_joint_position()
        #     current = np.linalg.norm([roll_angle, pitch_angle])
        #     trying += 1
        #     if (np.abs(target-current) > self.eps):
        #         break
        
        self.img = self.camera_object.capture_rgb()[60:420,140:500]
        # self.img = self.img.astype('uint8')
        r = 100.0 / self.img.shape[1]
        dim = (100, int(self.img.shape[0] * r))

        # self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.img = cv2.normalize(self.img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        ob = cv2.resize(self.img,dim,cv2.INTER_AREA)
        new_ob = cv2.subtract(ob, self.prev_obs)  # subtract previous frame from current 
        self.obs = self.set_obs_history(self.obs, new_ob, axis = self.axis)
        # self.done = not self.maze_object.check_collision(self.ball_object)
        self.prev_obs = ob
        self.done = True if len(self.maze_object.get_contact(self.ball_object))==0 else False
        
        #dt = time.time() - self.init_time
        #reward = np.exp(-np.abs(self.ball_object.check_distance(self.endDummy)))-np.exp(dt) if (self.maze_object.check_collision(self.ball_object)) else -1.0
        dist_to_goal = self.ball_object.check_distance(self.endDummy)
#         reward -= 10.0 
#         reward += 1.0/max(self.prev_dist_to_goal - dist_to_goal, 1)
        tt = max(self.prev_dist_to_goal - dist_to_goal, 1)
        reward = np.exp(-5*tt)
        # if (self.prev_dist_to_goal == dist_to_goal):
        if (np.abs(target_roll) == self.scene_parameter['scene-input-min-max'][1] or np.abs(target_pitch) == self.scene_parameter['scene-input-min-max'][1] or round(self.prev_dist_to_goal,4) == round(dist_to_goal,4)):
            self.local_stuck += 1
            if self.local_stuck>15: # terminate if stuck for 500 steps
                # self.done = True
                # when stuck generate random angle in deg the convert to radian
                ang = self.roll_joint_object.get_joint_position() + np.deg2rad(np.random.randn())
                self.roll_joint_object.set_joint_position(ang)
                ang = self.pitch_joint_object.get_joint_position() + np.deg2rad(np.random.randn())
                self.pitch_joint_object.set_joint_position(ang)
                self.pr.step()
                self.flag = "local_stuck_failed"
                # reward -= 100
                reward = -1
                self.local_stuck = 0
        self.prev_dist_to_goal = dist_to_goal
        if abs(dist_to_goal)<1.00e-2:
            # reward += 100
            reward = 1
            self.done = True
            self.flag = "succeed"
        info = {}
        self.last_reward = reward
        self.rec_video.append(ob)
        return self.obs, reward, self.done, info
        
    def reset(self):
        self.pr.stop()
        self.last_reward = 0
        # generate a new maze
        # add the shere
        #imageio.mimsave('lander_a2c.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
        if (self.verbose == 1) and (self.wall is not None) and len(self.rec_video)>4:
            imageio.mimsave('{}labyrinthGame{}_{}.gif'.format(self.log_location,time.time(),self.flag), [np.array(img) for i, img in enumerate(self.rec_video) if i%2 == 0], fps=5)
            self.rec_video = []
        if self.wall is not None:
            # remove
            self.wall.set_position(self.previous_wall.get_position())
            self.previous_wall.remove()
            self.wall.set_name('wall')
            self.wall.set_parent(Shape.get_object('roll_joint'))
            self.previous_wall = None
        self.wall = Object.get_object("wall")

        self.wall = self.wall.copy()
        self.wall.set_position([1,1,0])
        # self.reset_count+=1
        self.pr.start()
        # generating maze
        if self.difficulty_level==0:
            if (self.m is None):
                self._generate_maze()
            if self.reset_count<=0:
                self._generate_maze_start_end()
                self.reset_count = 1
                # fixed start & end
            pass
        elif self.difficulty_level==1:
            # fixed end & random start
            if (self.m is None):
                self._generate_maze()
            self._generate_maze_start_end()
            if self.reset_count<=0:
                self.temp_end = self.m.end
                self.reset_count = 1
            self.m.end = self.temp_end
            pass
        elif self.difficulty_level==2:
            # random maze 
            if (self.m is None):
                self._generate_maze()
                self._generate_maze_start_end()
        elif self.difficulty_level == 3:
            if (self.m is None):
                self._generate_maze()
                plane = Shape.get_object('Plane')
                color = np.random.randn(1,3).tolist()
                plane.set_color(color)
                self._generate_maze_start_end()
        else: # other levels to come
            self._generate_maze()
            self._generate_maze_start_end()


        length,breath = self.m.grid.shape
        m_copy = self.m.grid.copy()

        m_copy = np.delete(m_copy, (0,length-1),axis=1)
        m_copy = np.delete(m_copy, (0,length-1),axis=0)
        # m_copy
        length,breath = m_copy.shape

        play_area = (0.128, 0.128)
        h = 0.008
        board_thickness = 0.016
        max_mass = 150e-3 # 150g

        w, d = (play_area[0]/length), (play_area[0]/breath)
        z = h/2.0
        mazeBlocks = []
        hole = Shape.get_object("Hole")
        for x,rows in enumerate(m_copy):
            for y,cols in enumerate(rows):
                if(cols==0):
                    pos = [((x*w)+w/2.0), ((y*d)+d/2.0), z]
                    object = Shape.create(type=PrimitiveShape.CUBOID, 
                                        size=[w, d, h],
                                        position=[((x*w)+w/2.0), ((y*d)+d/2.0), z], mass=0.001)
                    if (x+1, y+1) == self.m.end:
                        (_, texture) = PyRep().create_texture('aruco_diction_joint_0.png')
                        object.set_texture(texture, TextureMappingMode.CUBE, uv_scaling=[w, d])
                        _.remove()
                        endDummy = Dummy.create(size=0.004)
                        endDummy.set_position(pos)
                    elif (x+1, y+1) == self.m.start:
                        startDummy = Dummy.create(size=0.004)
                        startDummy.set_position(pos)
                    mazeBlocks.append(object)
        #         print(cols)
                elif(cols==1):
                    hole_copy = hole.copy()
                    position=[((x*w)+w/2.0), ((y*d)+d/2.0), z]
                    hole_copy.set_position(position)
                    mazeBlocks.append(hole_copy)
                    
        groupedMazeBlock = self.pr.group_objects(mazeBlocks)
        endDummy.set_parent(groupedMazeBlock)
        startDummy.set_parent(groupedMazeBlock)
        # self.pr.step()

                # get the board frame location
        board_frame = object.get_object("wall")
        (board_frame_x, board_frame_y, board_frame_z) = board_frame.get_position()
        # print(board_frame_x, board_frame_y, board_frame_z)

        groupedMazeBlock.rotate([np.pi, 0, 0])
        min_x, min_y, min_z, max_x, max_y, max_z = board_frame.get_bounding_box()

        #move play_area to the main-boardframe

        groupedMazeBlock.set_position([board_frame_x, board_frame_y, (board_frame_z+(min_y-board_thickness)+z)])
        completeBoard = self.pr.group_objects([groupedMazeBlock,board_frame])
        # print(min_x, min_y, min_z, max_x, max_y, max_z)
        completeBoard.set_mass(max_mass)
        endDummy.set_parent(completeBoard)
        startDummy.set_parent(completeBoard)
        endDummy.set_name('endDummy')
        startDummy.set_name('startDummy')

        self.previous_wall = completeBoard
        ball_size = self.scene_parameter['scene-ball']
        brown = [150.0/256.0, 75.0/256.0, 0.0/256.0]
        self.ball_object = Shape.create(type=PrimitiveShape.SPHERE, 
                    size=[ball_size, ball_size, ball_size],
                    position=startDummy.get_position(), mass=0.001, color=brown)
        # groupedMazeBlock.set_orientation(board_frame.get_orientation())
        self.endDummy = endDummy
        self.startDummy = startDummy
        self.pr.step()

        self.roll_joint_object = Shape.get_object(self.scene_parameter['scene-input'][0])
        self.pitch_joint_object = Shape.get_object(self.scene_parameter['scene-input'][1])
        self.maze_object = Object.get_object(self.scene_parameter['scene-maze'])

        if (len(self.maze_object.get_contact(self.ball_object))==0):
            if (self.reset_count<=1):
                self.reset_count=0
            self.reset()
        self.init_time = time.time()

        self.camera_object = VisionSensor.get_object(self.scene_parameter['scene-output'])
        self.img = self.camera_object.capture_rgb()[60:420,140:500]
        # self.img = self.img.astype('uint8')
        r = 100.0 / self.img.shape[1]
        dim = (100, int(self.img.shape[0] * r))
        self.img = cv2.resize(self.img,dim,interpolation=cv2.INTER_AREA)
        self.img = cv2.normalize(self.img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
#         ob = self.img
        obs = self.build_obs_history(self.img.shape, self.sample, self.axis, self.img.dtype)
        self.obs = self.set_obs_history(obs, self.img, axis = self.axis)
        

        
        self.flag = "ball_fell_failed"
        self.roll_joint_object.set_joint_position(0)
        self.pitch_joint_object.set_joint_position(0)
        self.prev_obs = self.img
        self.reset_count+=1
        return self.obs
        
        
    def render(self, mode='human'):
        return self.img
    def close(self):
        self.pr.shutdown()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
#     ********************************Helper Functions********************************
    def build_obs_history(self, shape, n_sample = 3, axis=1, dtype=np.float32):
        temp = np.zeros(shape, dtype=dtype)
        obs = np.repeat(temp, repeats=n_sample, axis=axis)
        return obs
    def set_obs_history(self, obs, new_obs, axis=0):
        n_new_obs = new_obs.shape[axis]
        obs = np.roll(obs, n_new_obs, axis=axis)
        temp = np.delete(obs, np.s_[:n_new_obs], axis=axis)
        return np.concatenate([new_obs, temp], axis=axis)
    
    def _generate_maze_start_end(self):
        self.m.solver = BacktrackingSolver()
        self.m.generate_entrances(start_outer=False, end_outer=False)
        self.m.solve()
        pass
    
    def _generate_maze(self):
        self.m = Maze()
        self.m.generator = Prims(5, 5)
        self.m.generate()


