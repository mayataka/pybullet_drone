import pybullet_drone
import numpy as np

urdf_path = 'iris_description/iris.urdf'
time_step = 0.01

sim = pybullet_drone.DroneSimulator(urdf_path, time_step, log=True, record=True)
sim.init_simulation(simulation_name='test',
                    initial_time=0.0, 
                    initial_body_position=np.array([0, 0, 2.0]), 
                    initial_body_quaternion=np.array([0, 0, 0, 1.0]))

while True:
    u = 10 * np.random.rand(4)
    sim.step_simulation(u)
    t = sim.get_time()
    x = sim.get_state()
    print('t: ', t, ', x: ', x)