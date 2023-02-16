"""
Code for PEG-DA hydrogels

- with the model comprising:
    > Compressible neo-Hookean elasticity
    > diffusion of solvent:
        > Flory-type energetic and entropic chemical 
          free energy terms.

- with the numerical degrees of freedom:
    > vectorial mechanical displacement   
    > scalar pressure analog ( ln(Je)) : we use normalized value p/K \equiv ln(Je)
    > scalar chemical potential of solvent : we use normalized \mu/RT
    
- with basic units:
    > Length: mm
    >   Time:  s
    >   Mass: kg
    >  Moles: n-mol  
  and derived units
    > Pressure: kPa 
    > Force: mN
    
    Eric M. Stewart    and    Sooraj Narayan,   
   (ericstew@mit.edu)        (soorajn@mit.edu)     
    
                   Spring 2022 
                   
     - In writing this code, we drew on example code provided
       by Teng Zhang, Assistant Professor at Syracuse.
       (tzhang48@syr.edu)
    
If you use this in constructing your own code, please cite:
    • E. M. Stewart, S. Narayan, and L. Anand. On modeling the infiltration of water in a PEG-DA hydrogel
    and the resulting swelling under unconstrained and mechanically-constrained conditions. Extreme
    Mechanics Letters, 54:101775, July 2022.
    
"""

# Fenics-related packages
from dolfin import *
# Numerical array package
import numpy as np
# Plotting packages
import matplotlib.pyplot as plt
# Current time package
from datetime import datetime


# Set level of detail for log messages (integer)
# 
# Guide:
# CRITICAL  = 50, // errors that may lead to data corruption
# ERROR     = 40, // things that HAVE gone wrong
# WARNING   = 30, // things that MAY go wrong later
# INFO      = 20, // information of general interest (includes solver info)
# PROGRESS  = 16, // what's happening (broadly)
# TRACE     = 13, // what's happening (in detail)
# DBG       = 10  // sundry
#
set_log_level(30)

# The behavior of the form compiler FFC can be adjusted by prescribing
# various parameters. Here, we want to use some of the optimization
# features. ::

# Optimization options for the form compiler

parameters["form_compiler"]["cpp_optimize"] = True

ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "quadrature_degree": 4, \
               "precompute_ip_const": True}

# Define the solver parameters

newton_solver_parameters = {"nonlinear_solver": "newton",
                          "newton_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 50,
                                          "relative_tolerance": 1e-6,
                                          "absolute_tolerance": 1e-7,
                                          "report": True,
                                          "error_on_nonconvergence": False,
                                          "krylov_solver":
                                              {"nonzero_initial_guess": True,
                                               "monitor_convergence": True,
                                               "relative_tolerance": 1e-6,
                                               "absolute_tolerance": 1e-9,
                                               "divergence_limit": 1e50,
                                               "maximum_iterations": 2000,
                                              },
                                         },
                          }

'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''

# Create mesh and define function space
xDim = 0.8 #rod radius
yDim = 8.0 #rod length
# Last two numbers below are the number of elements in the two directions
mesh = RectangleMesh(Point(0, 0), Point(xDim, yDim), 8, 80)
x = SpatialCoordinate(mesh)

#Pick up on the boundary entities of the created mesh
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],yDim) and on_boundary
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],xDim) and on_boundary

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''

# Mechanical parameters
Gshear  = 1.e3                   # Shear modulus
Kbulk   = 10.0e3               # Bulk modulus
# Chemo-mechanical parameters
Omega   = 1.8e-5                 # fluid molecular volume
D       = 2.0e0                  # Diffusivity
RT      = 8.3145e-3*(273.0+25.0) # Gas constant
phi0    = 0.999                  # Initial polymer concentraiton
# Yasuda form parameters
alpha   = 7.7                    # shape parameter
gamma   = 3.e-4                  # numerical factor
# Flory-Huggins parameters 
chi0    = 0.52                   # zero-pressure mixing parameter
beta    = 1.9e-4                 # pressure-dependence slope

# linear pressure-dependence function for mixing param
def chi(pressure):
    return chi0 + beta*pressure


# Simulation time control-related params
t    = 0.0        # initialization of time
Ttot = 3600       # total simulation time 
ttd  = Ttot/100.0 # Decay time constant
dt   = 5.0        # Fixed step size

'''''''''''''''''''''
FEM SETUP
'''''''''''''''''''''

# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

TH = MixedElement([U2, P1, P1]) # Taylor-Hood style mixed element
ME = FunctionSpace(mesh, TH) # Total space for all DOFs

W2 = FunctionSpace(mesh, U2) # Vector space for visulization later
W = FunctionSpace(mesh,P1)   # Scalar space for visulization later

# Define test functions in weak form
(u_test, p_test,  mu_test) = TestFunctions(ME)    # Test function
dw = TrialFunction(ME) # Trial functions needed for automatic differentiation                           

# Define actual functions with the required DOFs
w = Function(ME)
u, p, mu = split(w)    # displacement, velocity, ~pressure,and chemcial potential of solvent

# A copy of functions to store values in last step for time-stepping.
w_old = Function(ME)
u_old,  p_old,  mu_old = split(w_old)   


# Initialize chemical potential, orresponding to nearly dry polymer.
mu0 = ln(1.0-phi0) + phi0 + chi(0.0)*phi0*phi0
init_mu = Constant(mu0)
mu_old = interpolate(init_mu,ME.sub(2).collapse())
assign(w_old.sub(2),mu_old)
assign(w.sub(2), mu_old)

'''''''''''''''''''''
HELPER FUNCTIONS
'''''''''''''''''''''

# Special gradient operators for Axisymmetric test functions 
#
# Gradient of vector field u   
def ax_grad_vector(u):
    grad_u = grad(u)
    return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, u[0]/x[0]]]) 
#
# Gradient of scalar field y
# (just need an extra zero for dimensions to work out)
def ax_grad_scalar(y):
    grad_y = grad(y)
    return as_vector([grad_y[0], grad_y[1], 0.])

# Axisymmetric deformation gradient 
def F_ax_calc(u):
    dim = len(u)
    Id = Identity(dim)          # Identity tensor
    F = Id + grad(u)            # 2D Deformation gradient
    F33 = (x[0]+u[0])/x[0]      # axisymmetric F33, R/R0    
    return as_tensor([[F[0,0], F[0,1], 0],
                  [F[1,0], F[1,1], 0],
                  [0, 0, F33]]) # Full axisymmetric F

# Polymer volume fraction \phi
def phi_calc(u,p):
    dfgrd_ax = F_ax_calc(u)  # = F
    detF = det(dfgrd_ax)     # = J
    detFe = exp(p)           # = Je
    return (detFe/detF*phi0) # phi = Je/J*phi_0

# Neo-Hookean Piola stress
def Piola(F,phi,p):
    return Gshear*( F - inv(F.T) ) + p/phi*Kbulk*inv(F.T) 

'''''''''''''''''''''''''''''
CONSTITUTIVE RELATIONS
'''''''''''''''''''''''''''''

# Kinematics
F_ax = F_ax_calc(u)
J = det(F_ax)  # Total volumetric jacobian
#
F_ax_old = F_ax_calc(u_old)         
J_old = det(F_ax_old)  # Total old volumetric jacobian
#
Ci = inv(F_ax.T*F_ax) # Inverse of the right Cauchy-Green stretch tensor

# We use the relation p = ln(Je)
# Elastic volumetric strain
Je = exp(p)                    
Je_old = exp(p_old)

# Polymer concentration
phi = Je/J*phi0
phi_old = Je_old/J_old*phi0

# Define Piola stress
TR = Piola(F_ax,phi,p)

# Calculate the Cauchy stress
T = (Piola(F_ax,phi,p)*F_ax.T)/J

# pressure for Flory-Huggins dependence
press = - (1.0/3.0)*tr(T)

# Yasuda term
fphi = exp(-alpha*(phi/(1.0-phi))) + gamma

# Calculate the mobility tensor
Mfluid = D/RT*((1/phi-1.)/Omega)*Ci*fphi

'''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''

# Time step field, constant within body
dk = Constant(dt)

# The weak form for the equilibrium equation
L0 = inner(1.0/Gshear*Piola(F_ax,phi,p),ax_grad_vector(u_test))*x[0]*dx

# The weak form for the scalar implicit equation for solvent chem. potential 
L1 = dot(p - ((ln(1.0-phi)+ phi + chi(press)*phi*phi)-mu)\
         *(1.0/Omega*RT/Kbulk),p_test)*x[0]*dx
    
# The weak form for mass balance of solvent         
L2 = dot((1.0/phi-1.0/phi_old),mu_test)*x[0]*dx \
     + dk*dot(Omega*Mfluid*ax_grad_scalar(mu)*RT,ax_grad_scalar(mu_test))*x[0]*dx
     
# Total weak form
L = L0 + L1 + L2 

# L0: Balance of forces (test fxn: u)
# L1: Chemical potential (test fxn: p)
# L2: Balance of mass (test fxn: mu)
    
# Automatic differentiation tangent:
a = derivative(L, w, dw)
   
'''''''''''''''''''''''
BOUNDARY CONDITIONS
'''''''''''''''''''''''      

# Mark boundary subdomians

# First, extract 1-D facets of mesh and use integer naming ("size_t")
facets = MeshFunction("size_t", mesh, 1) 
# Mark all boundaries with index '1'
DomainBoundary().mark(facets, 1)  
# Mark specific boundaries with next indices
Left().mark(facets, 2)
Bottom().mark(facets, 3)
Right().mark(facets, 4)
Top().mark(facets, 5)

# Boundary condition expressions as necessary
r = Expression(("mu0*exp(-t/td)"),
                mu0 = mu0, td = ttd, t = 0.0, degree=1)

# Boundary condition definitions
bcs_1 = DirichletBC(ME.sub(2), r, facets, 3)         # chem. pot. - Bottom
bcs_2 = DirichletBC(ME.sub(0).sub(0), 0, facets, 2)  # u1 fix - Left  
bcs_3 = DirichletBC(ME.sub(0).sub(1), 0, facets, 5)  # u2 fix - Top
bcs_4 = DirichletBC(ME.sub(0).sub(0), 0, facets, 5)  # u1 fix - Top

# BC sets for different steps of simulation
bcs = [bcs_1, bcs_2, bcs_3, bcs_4]

'''''''''''''''''''''
    RUN ANALYSIS
'''''''''''''''''''''

# Output file setup
file_results = XDMFFile("results/howon_swell.xdmf")
# "Flush_output" permits reading the output during simulation
# (Although this causes a minor performance hit)
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Store start time 
startTime = datetime.now()

print("------------------------------------")
print("Simulation Start")
print("------------------------------------")

# initialize an output array for tip displacement history
tipDisp  = np.zeros(1000)
time_out = np.zeros(1000) 
ii = 0

while (t < Ttot):

    # Report tip displacement each step 
    tipDisp[ii] = w_old(0,0)[1]
    time_out[ii] = t
    
    # Output storage
    #if ii%5 == 0: # This line can be used to limit frequency of output 
    if True:
        # Need this to see deformation
        w_old.rename("disp", "Displacement.")
        file_results.write(w_old.sub(0), t) 
        
        # Variable casting and renaming
        _w_1, _w_2,_w_3 = w_old.split() # Get DOFs from last step (or t=0 step)
        
        # Visualize \phi
        phi_Vis = phi_calc(_w_1,_w_2)
        phi_Viz = project(phi_Vis,W)
        phi_Viz.rename("phi", "Polymer volume fraction.")
        
        # Visualize chemical potential
        mu_Vis = _w_3*RT
        mu_Viz = project(mu_Vis,W)
        mu_Viz.rename("mu","Chemical potential.")
        
        # Visualize displacement
        u_Vis = _w_1
        u_Viz = project(u_Vis,W2)
        u_Viz.rename("disp","Displacement.")
        
        # Visualize pressure
        F_Vis = F_ax_calc(_w_1)
        T_Vis = (Piola(F_Vis,phi_Vis,_w_2)*F_Vis.T)/det(F_Vis)
        p_Vis = -(1.0/3.0)*tr(T_Vis)
        p_Viz = project(p_Vis,W)
        p_Viz.rename("pressure","Hydrostatic pressure.")
        
        # Visualize effective stretch
        C_Vis = F_Vis.T*F_Vis
        lambdaBar_Vis = sqrt(tr(C_Vis)/3.0)
        lambdaBar_Viz = project(lambdaBar_Vis,W)
        lambdaBar_Viz.rename("LambdaBar","Effective stretch.")
        
        # Visualize Je
        detFe_Vis = exp(_w_2)
        detFe_Viz = project(detFe_Vis,W)
        detFe_Viz.rename("Je","Elastic volume change.")
        
        # Write field quantities of interest
        file_results.write(u_Viz, t)
        file_results.write(p_Viz, t)
        file_results.write(mu_Viz, t)
        file_results.write(phi_Viz, t)
        file_results.write(detFe_Viz, t)
        file_results.write(lambdaBar_Viz, t)
    
    # update time variables in time-dependent BCs
    r.t = t
    
    # Set up the non-linear problem (free swell)
    GelProblem = NonlinearVariationalProblem(L, w, bcs, J=a,
                                           form_compiler_parameters=ffc_options)
 
    # Set up the non-linear solver
    solver  = NonlinearVariationalSolver(GelProblem)
    solver.parameters.update(newton_solver_parameters)

    # Solve the problem
    solver.solve()
    # Update DOFs for next step
    w_old.vector()[:] = w.vector()
    # increment time
    t += dt
    # increment counter
    ii += 1
    
    # Print progress of calculation periodically
    if t%200 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: Swell   |   Simulation Time: {}    |     Wallclock Time: {}".format(t, current_time))
        print()
        
        
# Report final tip disp value 
tipDisp[ii] = w_old(0,0)[1]
time_out[ii] = t

# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("------------------------------------")


'''''''''''''''''''''
    VISUALIZATION
'''''''''''''''''''''

font = {'size'   : 14}

plt.rc('font', **font)


# Experimental comparison
expData = np.genfromtxt('FreeSwellTip.csv', delimiter=',')
expData[:,1] = expData[:,1] - expData[0,1]

plt.figure(0)

plt.scatter(expData[:,0]/60.0, expData[:,1]/1e3, s=100,
                     edgecolors=(0.18588235, 0.32470588, 0.18588235,1),
                     color=(0.44611765, 0.77929412, 0.44611765,1),
                     label='Experiment', linewidth=3.0)

time_out = time_out[np.where(tipDisp!=0)]
tipDisp = tipDisp[np.where(tipDisp!=0)]
plt.plot(time_out/60.0, tipDisp, c='k',
                     label='Simulation', linewidth=5.0)

plt.legend(loc='upper right')
plt.xlabel('Time (min.)')
plt.ylabel(r'$u_{tip}$ (mm)')
#plt.xlim([4,7])

fig = plt.gcf()
fig.set_size_inches(6,4)
plt.tight_layout()
plt.savefig("free_swell.png", dpi=600)
