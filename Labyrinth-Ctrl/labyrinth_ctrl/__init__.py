from gym.envs.registration import register

register(
    id='labyrinthCtrl-v0',
    entry_point='labyrinth_ctrl.envs:LabyrinthCtrlEnv'
)

register(
    id='labyrinthCtrlDiscrete-v0',
    entry_point='labyrinth_ctrl.envs:LabyrinthCtrlDiscreteEnv'
)
register(
    id='labyrinthCtrlDiscreteNoIMG-v0',
    entry_point='labyrinth_ctrl.envs:LabyrinthCtrlDiscreteNoIMGEnv'
)