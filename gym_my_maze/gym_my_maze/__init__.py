from gym.envs.registration import register

register(
    id='mymaze-v0',
    entry_point='gym_my_maze.envs:MyMaze',
)
