import gym
from gym import error, spaces, utils
import numpy as np

class MyMaze(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.row = None
        self.col = None
        self.probe = None
        self.roadmap = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(2) 
        self.viewer = None

    def read(self, roadpath):
        roadmap = []
        with open(roadpath, 'r') as f:
            line = f.readline()
            row = [int(n) for n in line.split(' ')]
            roadmap.append(row)

    def padding(self, roadpath):
        roadmap = self.read(roadpath)
        self.row = len(roadmap)
        self.col = len(roadmap[0]) 
        new_roadmap = [1]*(self.col+2)
        for i in range(self.row):
            new_roadmap.append([1]+roadmap[i,:]+[1])
        new_roadmap.append([1]*(self.col+2))
        new_roadmap[1,1] = 0
        new_roadmap[self.row+1,self.col+1] = 0
        self.roadmap = new_roadmap

          
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        if action.index(max(action)) == 0:
            if self.roadmap[self.probe[0]+1,self.probe[1]] == 0:
                self.probe[0]+=1
        if action.index(max(action)) == 1:
            if self.roadmap[self.probe[0]-1,self.probe[1]] == 0:
                self.probe[0]-=1
        if action.index(max(action)) == 2:
            if self.roadmap[self.probe[0],self.probe[1]+1] == 0:
                self.probe[1]+=1
        if action.index(max(action)) == 3:
            if self.roadmap[self.probe[0],self.probe[1]-1] == 0:
                self.probe[1]-=1
        observation = np.array(self.probe)
        reward = self.row+1-self.probe[0]+self.col+1-self.probe[1]
        if self.probe == [self.row+1, self.col+1]:
            done = True
        else:
            done = False
        info = {}
        return observation, reward, done, info

    def reset(self,roadmap):
        self.padding(roadmap)
        self.probe = [1,1]

    def render(self, mode=render.modes):
        screen_width = 10*(self.col+2)
        screen_height = 10*(self.row+2)
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = 0, 10, 0, 10
            path = rendering.make_polygon([(l,b), (l,t), (r,t), (r,b)]).set_color([1,1,1])
            wall = rendering.make_polygon([(l,b), (l,t), (r,t), (r,b)]).set_color([0,0,0]) 
            for i in range(self.row+2):
                for j in range(self.col+2):
                    if self.roadmap[i,j] == 0:
                        self.viewer.add_geom(path.add_attr(translation=(10*j, 10*i)))
                    elif self.roadmap[i,j] == 1:
                        self.viewer.add_geom(wall.add_attr(translation=(10*j, 10*i)))
            probe_c = rendering.make_circle(radius = 4).set_color([1,0,0])
            if self.probe is None: return None
            else:
                self.viewer.add_onetime(probe_c.add_attr(translation=(1+10*self.probe[1], 1+10*self.probe[0])))
        return self.viewer.render(return_rgb_array= mode=='rgb_array')
              
    def close(self):
        utils.closer.Closer()
