import getopt, sys
from dqn.dqnnet import Dqnnet
from gym_my_maze.gym_my_maze.envs.gym_my_maze import MyMaze

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'r:c:v:', ['roadpath=','confpath=','videopath='])
    except getopt.GetoptError as err:
        print(err) 
        sys.exit()
    
    roadpath, confpath, videopath = None, None, None
    
    for o, a in opts:
        if o in ('-r', '--roadpath') and type(a)==str:
            roadpath = a
        elif o in ('-c', '--confpath') and type(a)==str:
            confpath = a 
        elif o in ('-v', '--videopath') and type(a)==str:
            videopath = a
        else:
            assert False, 'unhandled option' 
            
    if roadpath and videopath:
        MyMaze.padding(roadpath)
        Dqnnet().conf_settings(confpath)
        Dqnnet().create_env()
        Dqnnet().create_agent()
        Dqnnet().create_policy()
        Dqnnet().train()
        Dqnnet().create_video(videopath) 

if __name__ == "__main__":
    main()
