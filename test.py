from pybullet_drone import ModelInfo, DroneSimulator
import numpy as np

urdf_path = 'cf2_description/cf2x.urdf'
time_step = 0.01

model_info = ModelInfo(urdf_path=urdf_path, 
                       body_frame='center_of_mass_link', 
                       prop_frames=['prop0_link', 'prop1_link', 'prop2_link', 'prop3_link'])

sim = DroneSimulator(model_info, time_step)
sim.init(simulation_name='test',
         initial_time=0.0, 
         initial_body_position=np.array([0, 0, 2.0]), 
         initial_body_quaternion=np.array([0, 0, 0, 1.0]))

while True:
    u = 2 * np.random.rand(4)
    sim.step(u)
    t = sim.get_time()
    x = sim.get_state(velocity_reference_frame='local')
    print('t: ', t, ', x: ', x)