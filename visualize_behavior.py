import os
import sys
from generate_wcon import generate_wcon

duration = 2
dt = 0.005
logstep = 100

num_steps = int(duration/dt)
num_steps_logged = num_steps/logstep
rate = max(1,int(num_steps_logged/20.0))
#print("%s, %s"%(num_steps, rate))

name = sys.argv[1]
sim_dir = '/usr/people/chaudhry/hydra_scratch2/chaudhry/set8/{}/buffers/'.format(name)
# sim_dir = '/scratch/chaudhry/set8/{}/buffers/'.format(name)

# generate_wcon(os.path.join(sim_dir, 'worm_motion_log.txt'),
#                 os.path.join(sim_dir, 'worm_motion_log.wcon'),
#                 rate_to_plot=rate,
#                 plot=False,
#                 save_figure1_to=os.path.join(sim_dir, 'worm_motion_1.png'),
#                 save_figure2_to=os.path.join(sim_dir, 'worm_motion_2.png'),
#                 save_figure3_to=os.path.join(sim_dir, 'worm_motion_3.png'))

logstep = 5
num_steps = int(duration/dt)
num_steps_logged = num_steps/logstep
rate = max(1,int(num_steps_logged))

generate_wcon(os.path.join(sim_dir, 'worm_motion_log.txt'),
                os.path.join(sim_dir, 'worm_motion_log.wcon'),
                rate_to_plot=rate,
                plot=False,
                save_figure1_to=os.path.join(sim_dir, 'short_worm_motion_1.png'),
                save_figure2_to=os.path.join(sim_dir, 'short_worm_motion_2.png'),
                save_figure3_to=os.path.join(sim_dir, 'short_worm_motion_3.png'))
