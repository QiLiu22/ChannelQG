import sys
import numpy as np
import dedalus.public as d3
import h5py
import logging
logger = logging.getLogger(__name__)

# Numerics Parameters
Lx = 60
Ly = 30

Nx = 512
Ny = 256

dealias = 3/2
stop_sim_time = 5000
timestepper = d3.RK443
dtype = np.float64

#Physical Parameters
delx = Lx/Nx
# fine-scale dissipation coefficient
nu_2 = (delx)**2*4
res_cons = 0.2
# Ekman Damping coefficent
nu0 = float(sys.argv[1])/10
#beta 
beta = float(sys.argv[2])/10
# nu0 = 0.2

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.Chebyshev(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)

# Fields
#################
q1 = dist.Field(name='q1', bases=(xbasis,ybasis) )
q2 = dist.Field(name='q2', bases=(xbasis,ybasis) )
q1_Mean = dist.Field(name='q1_Mean', bases=ybasis )
q2_Mean = dist.Field(name='q2_Mean', bases=ybasis )

tau_q1_b = dist.Field(name='tau_q1_b', bases=xbasis)
tau_q1_t = dist.Field(name='tau_q1_t', bases=xbasis)
tau_q2_b = dist.Field(name='tau_q2_b', bases=xbasis)
tau_q2_t = dist.Field(name='tau_q2_t', bases=xbasis)

#################
Psi_1 = dist.Field(name='Psi_1', bases=(xbasis,ybasis) )
Psi_2 = dist.Field(name='Psi_2', bases=(xbasis,ybasis) )
tau_Psi_1 = dist.Field(name='tau_Psi_1')
tau_Psi_2 = dist.Field(name='tau_Psi_2')

tau_Psi_1_b = dist.Field(name='tau_Psi_1_b', bases=xbasis)
tau_Psi_1_t = dist.Field(name='tau_Psi_1_t', bases=xbasis)
tau_Psi_2_b = dist.Field(name='tau_Psi_2_b', bases=xbasis)
tau_Psi_2_t = dist.Field(name='tau_Psi_2_t', bases=xbasis)

tau_Psi_1bc_t = dist.Field(name='tau_Psi_1bc_t')
tau_Psi_1bc_b = dist.Field(name='tau_Psi_1bc_b')
tau_Psi_2bc_t = dist.Field(name='tau_Psi_2bc_t')
tau_Psi_2bc_b = dist.Field(name='tau_Psi_2bc_b')

# Substitutions
lift_basis = ybasis.derivative_basis(2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
lap = lambda A: d3.Laplacian(A)
xinteg = lambda A: d3.Integrate(A, ('x'))
integ = lambda A: d3.Integrate(A, ('x', 'y'))
xavg = lambda A: d3.Average(A, ('x'))
avg = lambda A: d3.Average(A, ('x', 'y'))

x, y = dist.local_grids(xbasis, ybasis)

J = lambda A, B: dx(A)*dy(B)-dy(A)*dx(B)

###
u_1 = -dy(Psi_1)
v_1 =  dx(Psi_1)
u_2 = -dy(Psi_2)
v_2 =  dx(Psi_2)
T = Psi_1-Psi_2
h2 = Psi_2-Psi_1

zeta_1 = -dy(u_1)+dx(v_1)
zeta_2 = -dy(u_2)+dx(v_2)

KE1 = avg(u_1**2+v_1**2)*0.5
KE2 = avg(u_2**2+v_2**2)*0.5
APE = avg(T**2)*0.5

u_1_mean = avg(u_1)
u_2_mean = avg(u_2)
Heat_Flux = avg((v_1+v_2)/2*(Psi_1-Psi_2))

# Problem
problem = d3.IVP([q1, q2, \
                  tau_q1_t, tau_q1_b, tau_q2_t, tau_q2_b, \
                  Psi_1, Psi_2, tau_Psi_1, tau_Psi_2, \
                  tau_Psi_1bc_t, tau_Psi_1bc_b, tau_Psi_2bc_t, tau_Psi_2bc_b, \
                  tau_Psi_1_t, tau_Psi_1_b, tau_Psi_2_t, tau_Psi_2_b
                    ], namespace=locals())

#################
problem.add_equation("dt(q1) - nu_2*lap(q1) +lift(tau_q1_b,-1)+lift(tau_q1_t,-2) = -res_cons*(xavg(q1)-q1_Mean) - (u_1*dx(q1)+v_1*dy(q1)) - beta*v_1")
problem.add_equation("dt(q2) - nu_2*lap(q2) +lift(tau_q2_b,-1)+lift(tau_q2_t,-2) + nu0*lap(Psi_2) = -res_cons*(xavg(q2)-q2_Mean) - (u_2*dx(q2)+v_2*dy(q2))- beta*v_2")
problem.add_equation("dy(q1)(y=-Ly/2)=0")
problem.add_equation("dy(q1)(y= Ly/2)=0")
problem.add_equation("dy(q2)(y=-Ly/2)=0")
problem.add_equation("dy(q2)(y= Ly/2)=0")

#################
problem.add_equation("lap(Psi_1)+(Psi_2-Psi_1)+lift(tau_Psi_1_b,-1)+lift(tau_Psi_1_t,-2)+tau_Psi_1=q1")
problem.add_equation("Psi_1(y= Ly/2)-tau_Psi_1bc_t=0");  problem.add_equation("Psi_1(y=-Ly/2)-tau_Psi_1bc_b=0")
problem.add_equation("xinteg(dy(Psi_1)(y=Ly/2)) = 0"); problem.add_equation("xinteg(dy(Psi_1)(y=-Ly/2)) = 0"); 

problem.add_equation("lap(Psi_2)+(Psi_1-Psi_2)+lift(tau_Psi_2_b,-1)+lift(tau_Psi_2_t,-2)+tau_Psi_2=q2")
problem.add_equation("Psi_2(y= Ly/2)-tau_Psi_2bc_t=0"); problem.add_equation("Psi_2(y=-Ly/2)-tau_Psi_2bc_b=0")
problem.add_equation("xinteg(dy(Psi_2)(y=Ly/2)) = 0");  problem.add_equation("xinteg(dy(Psi_2)(y=-Ly/2)) = 0")

problem.add_equation("integ(Psi_1)=0")
problem.add_equation("integ(Psi_2)=0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
q1_Mean['g'] = 10*np.tanh(y/5)+4*np.cosh(y/5)**(-2)*np.tanh(y/5)/5
q2_Mean['g'] = -10*np.tanh(y/5)

q1['g'] = 0; q2['g'] = 0
q1.fill_random('c', seed=100, distribution='normal', scale=1e-3) # Random noise
q1.low_pass_filter(shape=(10, 10)); q1.high_pass_filter(shape=(5, 5))

q1['g'] += q1_Mean['g']; q2['g'] += q2_Mean['g']

# Analysis
snapname = '2LayQG_channel_sp_%.1f_%d' %(nu0,Nx)
snapname = snapname.replace(".", "d" ); 
snapdata = solver.evaluator.add_file_handler(snapname, sim_dt=1, max_writes=10)
snapdata.add_task(-(-q1), name='q1')
snapdata.add_task(-(-q2), name='q2')
snapdata.add_task(-(-Psi_1), name='Psi_1')
snapdata.add_task(-(-Psi_2), name='Psi_2')
snapdata.add_task(-(-T), name='T')
snapdata.add_task(-(-zeta_1), name='zeta_1')
snapdata.add_task(-(-zeta_2), name='zeta_2')


diagname = '2LayQG_channel_dg_%.1f_%d' %(nu0,Nx)
diagname = diagname.replace(".", "d" ); 
diagdata = solver.evaluator.add_file_handler(diagname, sim_dt=0.1, max_writes=stop_sim_time*100)
diagdata.add_task(KE1, name='KE1')
diagdata.add_task(KE2, name='KE2')
diagdata.add_task(APE, name='APE')
diagdata.add_task(u_1_mean, name='u_1_mean')
diagdata.add_task(u_2_mean, name='u_2_mean')
diagdata.add_task(Heat_Flux, name='Heat_Flux')

# Flow properties
dt_change_freq = 10
flow_cfl = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
flow_cfl.add_property(abs(u_1), name='absu_1')
flow_cfl.add_property(abs(v_1), name='absv_1')
flow_cfl.add_property(abs(u_2), name='absu_2')
flow_cfl.add_property(abs(v_2), name='absv_2')

print_freq = 50
flow = d3.GlobalFlowProperty(solver, cadence=print_freq)
flow.add_property( (u_1**2+v_1**2)*0.5 , name='KE1')

###
# Main loop
timestep = 1e-7; 
delx = Lx/Nx; dely = Ly/Ny
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)

        if (solver.iteration) % dt_change_freq == 0:
            maxU = max(1e-10,flow_cfl.max('absu_1'),flow_cfl.max('absu_2')); maxV = max(1e-10,flow_cfl.max('absv_1'),flow_cfl.max('absv_2'))
            timestep_CFL = min(delx/maxU,dely/maxV)*0.2
            timestep = min(max(1e-10, timestep_CFL), 0.1)

        if (solver.iteration) % print_freq == 0:
            logger.info('Iteration=%i, Time=%.3f, dt=%.3e, KE1=%.3f' %(solver.iteration, solver.sim_time, timestep, flow.volume_integral('KE1')))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
