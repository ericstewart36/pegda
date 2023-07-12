"""
Code for PEG-DA hydrogels

- with the model comprising:
    > Compressible neo-Hookean elasticity
    > diffusion of solvent:
        > Flory-type energetic and entropic chemical 
          free energy terms.

- with the numerical degrees of freedom:
    > vectorial mechanical displacement   
    > scalar chemical potential of solvent : we use normalized  mu = mu/RT
    > species concentration:  we use normalized  c= Omega*cmat
    
Units:
> Length: mm
> Time:  s
> Mass: tonne (1000 kg)
> Force: N
> Pressure: MPa 
> Energy: mJ
> Species concentration: mol/mm^3
> Molar volume: mm^3/mol
> Chemical potential: mJ/mol
> Species Diffusivity: mm^2/s
> Gas Constant 8.3145E3 mJ/(mol K)

UPDATED: Summer 2023 in collaboration with Prof. Lallit Anand:
  - We now use the concentration as an auxiliary variable rather than ln(Je), 
 since we view that the concentration-based approach is more straightforward.
 

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

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 4      

'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''
# Create mesh 
xDim = 0.8 #rod radius, mm
yDim = 8.0 #rod length, mm
# Last two numbers below are the number of elements in the two directions
mesh = RectangleMesh(Point(0, 0), Point(xDim, yDim), 8, 80)

# This says "spatial coordinates" but is really the referential coordinates,
# since the mesh does not convect in FEniCS.
x = SpatialCoordinate(mesh) 

# Identify the boundary entities of the created mesh
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

# Identify 1-D boundary subdomains of mesh and use integer naming ("size_t")
facets = MeshFunction("size_t", mesh, 1) 
# Mark all boundaries with index '1'
DomainBoundary().mark(facets, 1)  
# Mark specific boundaries with next indices
Left().mark(facets, 2)
Bottom().mark(facets, 3)
Right().mark(facets, 4)
Top().mark(facets, 5)

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''

# Mechanical parameters
Gshear  = 1                  # Shear modulus
Kbulk   = 10.0                 # Bulk modulus
# Chemo-mechanical parameters
Omega   = 1.8e4                 # fluid molecular volume
D       = 2.0e0                  # Diffusivity
RT      = 8.3145e3*(273.0+25.0) # Gas constant
phi0    = 0.999                  # Initial polymer concentraiton
# Yasuda form parameters
alpha   = 7.7                    # shape parameter
gamma   = 3.e-4                  # numerical factor
# Flory-Huggins parameters 
chi0    = 0.52                   # zero-pressure mixing parameter
beta    = 1.9e-1                 # pressure-dependence slope

# linear pressure-dependence function for mixing param
def chi(pressure):
    return chi0 + beta*pressure

# Simulation time control-related params
t    = 0.0        # initialization of time
Ttot = 36000      # total simulation time 
ttd  = Ttot/100.0 # Decay time constant
dt   = 100.0      # Fixed step size

'''''''''''''''''''''
FEM SETUP
'''''''''''''''''''''

# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # For displacement
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For  normalized chemical potential and  normalized species concentration
#
TH = MixedElement([U2, P1, P1]) # Taylor-Hood style mixed element
ME = FunctionSpace(mesh, TH)    # Total space for all DOFs

# Define actual functions with the required DOFs
w = Function(ME)
u, mu, c = split(w)  # displacement u, chemical potential  mu,  concentration c

# A copy of functions to store values in the previous step for time-stepping
w_old = Function(ME)
u_old,  mu_old,  c_old = split(w_old)   

# Define test functions in 
w_test = TestFunction(ME)                
u_test, mu_test, c_test = split(w_test)   

#Define trial functions neede for automatic differentiation
dw = TrialFunction(ME)             

# Initialize chemical potential, orresponding to nearly dry polymer.
mu0 = ln(1.0-phi0) + phi0 + chi0*phi0*phi0
init_mu = Constant(mu0) 
mu_init = interpolate(init_mu,ME.sub(1).collapse())
assign(w_old.sub(1),mu_init)
assign(w.sub(1), mu_init)

# Assign initial species normalized concentration c0
c0 = 1/phi0 - 1
init_c = Constant(c0)
c_init = interpolate(init_c, ME.sub(2).collapse())
assign(w_old.sub(2),c_init)
assign(w.sub(2), c_init)

'''''''''''''''''''''
Subroutines
'''''''''''''''''''''
# Special gradient operators for Axisymmetric test functions 
#
# Gradient of vector field u   
def ax_grad_vector(u):
    grad_u = grad(u)
    return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, u[0]/x[0]]]) 

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

#  Elastic Je
def Je_calc(u,c):
    F_ax = F_ax_calc(u)      # = F
    detF = det(F_ax)         # = J
    #
    detFs = 1.0 + c          # = Js
    Je    = (detF/detFs)     # = Je
    return   Je    

# Normalized Piola stress for Neo-Hookean material
def Piola_calc(F,c,Je):
    Tmat = (F - inv(F.T) ) + (1+c)*(Kbulk/Gshear)*ln(Je)*inv(F.T) 
    return Tmat

# Species flux
def Flux_calc(u, mu, c):
    F_ax = F_ax_calc(u) 
    #
    Cinv = inv(F_ax.T*F_ax) 
    #
    phi = 1/(1+c)
    # Yasuda term
    fphi = exp(-alpha*(phi/(1.0-phi))) + gamma
    # Mobility tensor
    Mob = (D*c)/(Omega*RT)*Cinv*fphi
    # (Referential) flux vector
    Jmat = - RT* Mob * ax_grad_scalar(mu)
    return Jmat

'''''''''''''''''''''''''''''
Kinematics and Constitutive relations
'''''''''''''''''''''''''''''
F_ax = F_ax_calc(u)
J = det(F_ax)  # Total volumetric jacobian

# Elastic volumetric Jacobian
Je     = Je_calc(u,c)                    
Je_old = Je_calc(u_old,c_old)

#  Normalized Piola stress
Tmat = Piola_calc(F_ax, c, Je)

# Normalized species flux
Jmat = Flux_calc(u, mu, c)

# Pressure for pressure-dependent Flory-Huggins mixing parameter
press = -(1/3)*tr((Gshear*Tmat*F_ax.T)/J)

'''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Balance of forces (test fxn: u)
# Res_1: Balance of mass   (test fxn: mu)
# Res_2: Auciliary variable (test fxn: c)

# Time step field, constant  
dk = Constant(dt)

# The weak form for the equilibrium equation
Res_0 = inner(Tmat, ax_grad_vector(u_test) )*x[0]*dx

# The weak form for the mass balance of solvent      
Res_1 = dot((c - c_old)/dk, mu_test)*x[0]*dx \
        -  Omega*dot(Jmat , ax_grad_scalar(mu_test) )*x[0]*dx
 
# The weak form for the concentration
fac = 1/(1+c)

fac1 =  mu - ( ln(1.0-fac)+ fac + chi(press)*fac*fac)

fac2 = (Omega*Kbulk/RT)*ln(Je)  

fac3 = fac1 + fac2 

Res_2 = dot(fac3, c_test)*x[0]*dx
     
# Total weak form
Res = Res_0 + Res_1 + Res_2 

# Automatic differentiation tangent:
a = derivative(Res, w, dw)
   
'''''''''''''''''''''''
BOUNDARY CONDITIONS
'''''''''''''''''''''''      
# Boundary condition expressions as necessary
r = Expression(("mu0*exp(-t/td)"),
                mu0 = mu0, td = ttd, t = 0.0, degree=1)

# Boundary condition definitions
bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 2)    # u1 fix - left  
bcs_2 = DirichletBC(ME.sub(0).sub(0), 0, facets,4) # u1 fix - Right
bcs_3 = DirichletBC(ME.sub(0), Constant((0,0)), facets, 5)     # u1/u2 fix - top
bcs_4 = DirichletBC(ME.sub(1), r, facets, 3)          # chem. pot. - bottom

# BCs set 
bcs = [bcs_1, bcs_2, bcs_3, bcs_4]

'''''''''''''''''''''
Define nonlinear problem
'''''''''''''''''''''
GelProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
solver  = NonlinearVariationalSolver(GelProblem)

#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance'] = 1.e-8
prm['newton_solver']['relative_tolerance'] = 1.e-8
prm['newton_solver']['maximum_iterations'] = 30

'''''''''''''''''''''
Set-up output files
'''''''''''''''''''''
# Output file setup
file_results = XDMFFile("results/Howon_const_swell.xdmf")
# "Flush_output" permits reading the output during simulation
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Function space for projection of results
W2 = FunctionSpace(mesh, U2)   # Vector space for visualization  
W  = FunctionSpace(mesh,P1)    # Scalar space for visualization 

# Subroutine for writing output to file
def writeResults(t):
    
    # Variable casting and renaming
    _w_1, _w_2, _w_3 = w_old.split() # Get DOFs from last step (or t=0 step)
       
    # Visualize displacement
    u_Vis = _w_1
    u_Viz = project(u_Vis, W2)
    u_Viz.rename("disp"," ")
    
    # Visualize  normalized chemical potential
    mu_Vis = _w_2 
    mu_Viz = project(mu_Vis,W)
    mu_Viz.rename("mu"," ")
    
    # Visualize  normalized concentration
    c_Vis = _w_3 
    c_Viz = project(c_Vis,W)
    c_Viz.rename("c"," ")
    
    # Visualize Je
    detFe_Vis =  Je_calc(_w_1, _w_3)
    detFe_Viz =  project(detFe_Vis,W)
    detFe_Viz.rename("Je"," ")
    
    # Visualize phi
    phi_Vis = 1/(1 + _w_3)
    phi_Viz = project(phi_Vis,W)
    phi_Viz.rename("phi", " ")
    
    # Visualize effective stretch
    F_Vis = F_ax_calc(_w_1)
    C_Vis = F_Vis.T*F_Vis
    lambdaBar_Vis = sqrt(tr(C_Vis)/3.0)
    lambdaBar_Viz = project(lambdaBar_Vis,W)
    lambdaBar_Viz.rename("LambdaBar"," ")
    
    # Visualize J
    J_Vis = det(F_Vis)
    J_Viz = project(J_Vis,W)
    J_Viz.rename("J"," ")    
    
    # Write field quantities of interest
    file_results.write(u_Viz, t)
    file_results.write(mu_Viz, t)
    file_results.write(c_Viz, t)
    file_results.write(detFe_Viz, t)
    file_results.write(phi_Viz, t)
    file_results.write(lambdaBar_Viz, t)
    file_results.write(J_Viz, t)    
    
# Write initial values
writeResults(t=0.0)


print("------------------------------------")
print("Simulation Start")
print("------------------------------------")
# Store start time 
startTime = datetime.now()

# initialize counter for print out of progress
ii = 0

# initialize an output array for tip displacement history
tipDisp  = np.zeros(1000)
time_out = np.zeros(1000) 

while (t < Ttot):
    
    # increment time
    t += dt
    
    # increment counter
    ii += 1

    # Update time variables in time-dependent BCs
    r.t = t

    # Solve the problem
    (iter, converged) = solver.solve()  
    
    # Report tip displacement each step 
    tipDisp[ii] = w_old(0,0)[1]
    time_out[ii] = t
    
    # Update DOFs for next step
    w_old.vector()[:] = w.vector()

    # Write output to *.xdmf file at time t
    writeResults(t) 
    
    # Print progress of calculation periodically
    if t%100 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: Swell   |   Simulation Time: {}    |     Wallclock Time: {}".format(t, current_time))
        print("Iterations: {}".format(iter))
        print()
        
print("--------------------------------------------")
print("End computation")    
# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("-------------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("-------------------------------------------")

# Report final tip disp value 
tipDisp[ii] = w_old(0,0)[1]
time_out[ii] = t

'''''''''''''''''''''
    VISUALIZATION
'''''''''''''''''''''

font = {'size'   : 14}
plt.rc('font', **font)

# Experimental comparison
expData = np.genfromtxt('ConstrSwellTip.csv', delimiter=',')
expData[:,1] = expData[:,1] - expData[0,1]

              
plt.figure(0)
plt.scatter(expData[:,0], expData[:,1]*1e-3, s=50,
                     edgecolors=(0.15, 0.35, 0.6,1),
                     color=(0.25, 0.625, 1.0,1),
                     label='Experiment', linewidth=1.0)

time_out = time_out[np.where(tipDisp!=0)]
tipDisp = tipDisp[np.where(tipDisp!=0)]
plt.plot(time_out/60.0, tipDisp, c='k',
                     label='Simulation', linewidth=2.0)

plt.legend(loc='upper right')
plt.xlabel('Time (min.)')
plt.ylabel(r'$u_{tip}$ (mm)')

fig = plt.gcf()
fig.set_size_inches(6,4)
plt.tight_layout()
plt.savefig("constr_swell.png", dpi=600)
