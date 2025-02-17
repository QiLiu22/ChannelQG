# python3 (initial U mean guess)
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
stop_sim_time = 50000
timestepper = d3.RK443
dtype = np.float64

#Physical Parameters
delx = Lx/Nx
nu2 = (delx)**2*4
T = 0.2
lambda_scale = Lx / 50 / (2 * np.pi)
Delu = float(sys.argv[1])/100
r = 0.16 * (Delu / lambda_scale)
beta = 0.05
k_d = 1


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
yinteg = lambda A: d3.Integrate(A, ('y'))
integ = lambda A: d3.Integrate(A, ('x', 'y'))
xavg = lambda A: d3.Average(A, ('x'))
yavg = lambda A: d3.Average(A, ('y'))
avg = lambda A: d3.Average(A, ('x', 'y'))

x, y = dist.local_grids(xbasis, ybasis)

J = lambda A, B: dx(A)*dy(B)-dy(A)*dx(B)

###
u1 = -dy(Psi_1)
U1 = xavg(u1)
u1_prime = u1-U1
v1 =  dx(Psi_1)
V1 = xavg(v1)
v1_prime = v1-V1
u2 = -dy(Psi_2)
U2 = xavg(u2)
u2_prime = u2-U2
v2 =  dx(Psi_2)
V2 = xavg(v2)
v2_prime = v2-V2
Psi = Psi_1-Psi_2


zeta_1 = -dy(u1)+dx(v1)
zeta_2 = -dy(u2)+dx(v2)

KE1 = integ(u1**2+v1**2)*0.5
KE2 = integ(u2**2+v2**2)*0.5
APE = integ((Psi**2)*k_d**2/4)
Relaxation_E = 0

# Problem
problem = d3.IVP([q1, q2, \
                  tau_q1_t, tau_q1_b, tau_q2_t, tau_q2_b, \
                  Psi_1, Psi_2, tau_Psi_1, tau_Psi_2, \
                  tau_Psi_1bc_t, tau_Psi_1bc_b, tau_Psi_2bc_t, tau_Psi_2bc_b, \
                  tau_Psi_1_t, tau_Psi_1_b, tau_Psi_2_t, tau_Psi_2_b
                    ], namespace=locals())

#################
problem.add_equation("dt(q1) - nu2*lap(q1) +lift(tau_q1_b,-1)+lift(tau_q1_t,-2) = -T*(xavg(q1)-Q1M) - (u1*dx(q1)+v1*dy(q1)) - beta * v1")
problem.add_equation("dt(q2) - nu2*lap(q2) +lift(tau_q2_b,-1)+lift(tau_q2_t,-2) + r*lap(Psi_2) = -T*(xavg(q2)-Q2M) - (u2*dx(q2)+v2*dy(q2)) - beta * v2")
problem.add_equation("dy(q1)(y=-Ly/2)=0")
problem.add_equation("dy(q1)(y= Ly/2)=0")
problem.add_equation("dy(q2)(y=-Ly/2)=0")
problem.add_equation("dy(q2)(y= Ly/2)=0")

#################
problem.add_equation("lap(Psi_1)- k_d**2 / 2 * (Psi_1-Psi_2)+lift(tau_Psi_1_b,-1)+lift(tau_Psi_1_t,-2)+tau_Psi_1=q1")
problem.add_equation("Psi_1(y= Ly/2)-tau_Psi_1bc_t=0");  problem.add_equation("Psi_1(y=-Ly/2)-tau_Psi_1bc_b=0")
problem.add_equation("xinteg(dy(Psi_1)(y=Ly/2)) = 0"); problem.add_equation("xinteg(dy(Psi_1)(y=-Ly/2)) = 0"); 

problem.add_equation("lap(Psi_2)+ k_d**2 / 2 * (Psi_1-Psi_2)+lift(tau_Psi_2_b,-1)+lift(tau_Psi_2_t,-2)+tau_Psi_2=q2")
problem.add_equation("Psi_2(y= Ly/2)-tau_Psi_2bc_t=0"); problem.add_equation("Psi_2(y=-Ly/2)-tau_Psi_2bc_b=0")
problem.add_equation("xinteg(dy(Psi_2)(y=Ly/2)) = 0");  problem.add_equation("xinteg(dy(Psi_2)(y=-Ly/2)) = 0")

problem.add_equation("integ(Psi_1)=0")
problem.add_equation("integ(Psi_2)=0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
Q1M['g'] = (10*np.tanh(y/5)+4*np.cosh(y/5)**(-2)*np.tanh(y/5)/5)/5
Q2M['g'] = (-10*np.tanh(y/5))/5

q1['g'] = 0; q2['g'] = 0
q1.fill_random('c', seed=100, distribution='normal', scale=1e-3) # Random noise
q1.low_pass_filter(shape=(10, 10)); q1.high_pass_filter(shape=(5, 5))

q1['g'] += Q1M['g']; q2['g'] += Q2M['g']

q1_bar = xavg(q1)  # Zonal mean of q1
q1_prime = q1 - q1_bar
q2_bar = xavg(q2)  # Zonal mean of q2
q2_prime = q2 - q2_bar

Psi_1_bar = xavg(Psi_1)
Psi_1_prime = Psi_1 - Psi_1_bar
Psi_2_bar = xavg(Psi_2)
Psi_2_prime = Psi_2 - Psi_2_bar

HF1 = -dy(xavg(v1_prime * q1_prime))
HF2 = -dy(xavg(v2_prime * q2_prime))
          
ZKE1 = yinteg(U1**2+V1**2)*0.5
ZKE2 = yinteg(U2**2+V2**2)*0.5
ZAPE = yinteg(((Psi_1_bar-Psi_2_bar)**2)*k_d**2/4)
EKE1 = integ(u1_prime**2+v1_prime**2)*0.5
EKE2 = integ(u2_prime**2+v2_prime**2)*0.5
EAPE = integ(((Psi_1_prime-Psi_2_prime)**2)*k_d**2/4)

H = xavg(Psi)
h = Psi - H
EHF = (v1_prime + v2_prime) * h
EFH = avg(EHF)

b = dy(Psi)
TG = avg(b)
# Analysis
snapname = '2LayQG_channel_snap_' + 'U' + sys.argv[1]
snapdata = solver.evaluator.add_file_handler(snapname, sim_dt=1, max_writes=100)
snapdata.add_task(-(-q1), name='q1')
snapdata.add_task(-(-q2), name='q2')
snapdata.add_task(-(-zeta_1), name='zeta_1')
snapdata.add_task(-(-zeta_2), name='zeta_2')



diagname = '2LayQG_channel_diag_' + 'U' + sys.argv[1]
diagdata = solver.evaluator.add_file_handler(diagname, sim_dt=0.1, max_writes=stop_sim_time*100)
diagdata.add_task(KE1, name='KE1')
diagdata.add_task(KE2, name='KE2')
diagdata.add_task(APE, name='APE')
diagdata.add_task(ZKE1, name='ZKE1')
diagdata.add_task(ZKE2, name='ZKE2')
diagdata.add_task(EKE1, name='EKE1')
diagdata.add_task(EKE2, name='EKE2')
diagdata.add_task(ZAPE, name='ZAPE')
diagdata.add_task(EAPE, name='EAPE')
diagdata.add_task(EFH, name='EFH')
diagdata.add_task(TG, name='TG')


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
