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
nu2 = (delx)**2*4
res_cons = 0.2
nu0 = float(sys.argv[1])/10
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
Q1M = dist.Field(name='Q1M', bases=ybasis )
Q2M = dist.Field(name='Q2M', bases=ybasis )

tau_q1_b = dist.Field(name='tau_q1_b', bases=xbasis)
tau_q1_t = dist.Field(name='tau_q1_t', bases=xbasis)
tau_q2_b = dist.Field(name='tau_q2_b', bases=xbasis)
tau_q2_t = dist.Field(name='tau_q2_t', bases=xbasis)

#################
P0_1 = dist.Field(name='P0_1', bases=(xbasis,ybasis) )
P0_2 = dist.Field(name='P0_2', bases=(xbasis,ybasis) )
tau_P0_1 = dist.Field(name='tau_P0_1')
tau_P0_2 = dist.Field(name='tau_P0_2')

tau_P0_1_b = dist.Field(name='tau_P0_1_b', bases=xbasis)
tau_P0_1_t = dist.Field(name='tau_P0_1_t', bases=xbasis)
tau_P0_2_b = dist.Field(name='tau_P0_2_b', bases=xbasis)
tau_P0_2_t = dist.Field(name='tau_P0_2_t', bases=xbasis)

tau_P0_1bc_t = dist.Field(name='tau_P0_1bc_t')
tau_P0_1bc_b = dist.Field(name='tau_P0_1bc_b')
tau_P0_2bc_t = dist.Field(name='tau_P0_2bc_t')
tau_P0_2bc_b = dist.Field(name='tau_P0_2bc_b')

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
u1 = -dy(P0_1)
v1 =  dx(P0_1)
u2 = -dy(P0_2)
v2 =  dx(P0_2)
h1 = P0_1-P0_2
h2 = P0_2-P0_1

zeta_1 = -dy(u1)+dx(v1)
zeta_2 = -dy(u2)+dx(v2)

KE1 = avg(u1**2+v1**2)*0.5
KE2 = avg(u2**2+v2**2)*0.5
PE1 = avg(h1**2)*0.5
PE2 = avg(h2**2)*0.5
U1_mean = avg(u1)
U2_mean = avg(u2)
vt_mean = avg((v1+v2)/2*(P0_1-P0_2))

# Problem
problem = d3.IVP([q1, q2, \
                  tau_q1_t, tau_q1_b, tau_q2_t, tau_q2_b, \
                  P0_1, P0_2, tau_P0_1, tau_P0_2, \
                  tau_P0_1bc_t, tau_P0_1bc_b, tau_P0_2bc_t, tau_P0_2bc_b, \
                  tau_P0_1_t, tau_P0_1_b, tau_P0_2_t, tau_P0_2_b
                    ], namespace=locals())

#################
problem.add_equation("dt(q1) - nu2*lap(q1) +lift(tau_q1_b,-1)+lift(tau_q1_t,-2) = -res_cons*(xavg(q1)-Q1M) - (u1*dx(q1)+v1*dy(q1))")
problem.add_equation("dt(q2) - nu2*lap(q2) +lift(tau_q2_b,-1)+lift(tau_q2_t,-2) + nu0*lap(P0_2) = -res_cons*(xavg(q2)-Q2M) - (u2*dx(q2)+v2*dy(q2))")
problem.add_equation("dy(q1)(y=-Ly/2)=0")
problem.add_equation("dy(q1)(y= Ly/2)=0")
problem.add_equation("dy(q2)(y=-Ly/2)=0")
problem.add_equation("dy(q2)(y= Ly/2)=0")

#################
problem.add_equation("lap(P0_1)+(P0_2-P0_1)+lift(tau_P0_1_b,-1)+lift(tau_P0_1_t,-2)+tau_P0_1=q1")
problem.add_equation("P0_1(y= Ly/2)-tau_P0_1bc_t=0");  problem.add_equation("P0_1(y=-Ly/2)-tau_P0_1bc_b=0")
problem.add_equation("xinteg(dy(P0_1)(y=Ly/2)) = 0"); problem.add_equation("xinteg(dy(P0_1)(y=-Ly/2)) = 0"); 

problem.add_equation("lap(P0_2)+(P0_1-P0_2)+lift(tau_P0_2_b,-1)+lift(tau_P0_2_t,-2)+tau_P0_2=q2")
problem.add_equation("P0_2(y= Ly/2)-tau_P0_2bc_t=0"); problem.add_equation("P0_2(y=-Ly/2)-tau_P0_2bc_b=0")
problem.add_equation("xinteg(dy(P0_2)(y=Ly/2)) = 0");  problem.add_equation("xinteg(dy(P0_2)(y=-Ly/2)) = 0")

problem.add_equation("integ(P0_1)=0")
problem.add_equation("integ(P0_2)=0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
Q1M['g'] = 10*np.tanh(y/5)+4*np.cosh(y/5)**(-2)*np.tanh(y/5)/5
Q2M['g'] = -10*np.tanh(y/5)

q1['g'] = 0; q2['g'] = 0
q1.fill_random('c', seed=100, distribution='normal', scale=1e-3) # Random noise
q1.low_pass_filter(shape=(10, 10)); q1.high_pass_filter(shape=(5, 5))

q1['g'] += Q1M['g']; q2['g'] += Q2M['g']

# Analysis
snapname = '2LayQG_channel_sp_%.1f_%d' %(nu0,Nx)
snapname = snapname.replace(".", "d" ); 
snapdata = solver.evaluator.add_file_handler(snapname, sim_dt=1, max_writes=10)
snapdata.add_task(-(-q1), name='q1')
snapdata.add_task(-(-q2), name='q2')
snapdata.add_task(-(-P0_1), name='P0_1')
snapdata.add_task(-(-P0_2), name='P0_2')
snapdata.add_task(-(-h1), name='h1')
snapdata.add_task(-(-h2), name='h2')
snapdata.add_task(-(-zeta_1), name='zeta_1')
snapdata.add_task(-(-zeta_2), name='zeta_2')


diagname = '2LayQG_channel_dg_%.1f_%d' %(nu0,Nx)
diagname = diagname.replace(".", "d" ); 
diagdata = solver.evaluator.add_file_handler(diagname, sim_dt=0.1, max_writes=stop_sim_time*100)
diagdata.add_task(KE1, name='KE1')
diagdata.add_task(KE2, name='KE2')
diagdata.add_task(PE1, name='PE1')
diagdata.add_task(PE2, name='PE2')
diagdata.add_task(U1_mean, name='U1_mean')
diagdata.add_task(U2_mean, name='U2_mean')
diagdata.add_task(vt_mean, name='vt_mean')

# Flow properties
dt_change_freq = 10
flow_cfl = d3.GlobalFlowProperty(solver, cadence=dt_change_freq)
flow_cfl.add_property(abs(u1), name='absu1')
flow_cfl.add_property(abs(v1), name='absv1')
flow_cfl.add_property(abs(u2), name='absu2')
flow_cfl.add_property(abs(v2), name='absv2')

print_freq = 50
flow = d3.GlobalFlowProperty(solver, cadence=print_freq)
flow.add_property( (u1**2+v1**2)*0.5 , name='KE1')

###
# Main loop
timestep = 1e-7; 
delx = Lx/Nx; dely = Ly/Ny
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)

        if (solver.iteration) % dt_change_freq == 0:
            maxU = max(1e-10,flow_cfl.max('absu1'),flow_cfl.max('absu2')); maxV = max(1e-10,flow_cfl.max('absv1'),flow_cfl.max('absv2'))
            timestep_CFL = min(delx/maxU,dely/maxV)*0.2
            timestep = min(max(1e-10, timestep_CFL), 0.1)

        if (solver.iteration) % print_freq == 0:
            logger.info('Iteration=%i, Time=%.3f, dt=%.3e, KE1=%.3f' %(solver.iteration, solver.sim_time, timestep, flow.volume_integral('KE1')))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
