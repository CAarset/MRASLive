import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from ngsolve import *
from ngsolve.webgui import Draw
from pathlib import Path
import ipywidgets as widgets

#function that takes a 2D list and maps it onto a gridfunction
#one could use uniform grids as well, but this is less flexible
def to_gridfunc (list_2D, fes, mesh):
    array = (np.array(list_2D)).astype(np.float64)
    array_v = VoxelCoefficient((-2, -1), (2, 1), array, linear=True)
    gf = GridFunction(fes)
    gf.Set(array_v)
    return gf

#this function also brings a list into a gridfunction while ensuring, that the bnd is 0 by bringing it down to 0 in a bnd strip
def to_gridfunc_0 (list_2D, fes, mesh):
    array = (np.array(list_2D)).astype(np.float64)
    array_v = VoxelCoefficient((-2, -1), (2, 1), array, linear=True)
    gf = GridFunction(fes)
    gf.Set(array_v*IfPos((x-.05), 1, x*20)*IfPos((y-.05), 1, y*20)*IfPos((0.95-x), 1, 20*(1-x))*IfPos((0.95-y), 1, 20*(1-y)))
    return gf


# computes L2 norm
def norm(q,mesh):
    return Integrate(q**2,mesh)**(1/2) #change if q is not real values anymore

# adds noise to a gridfunction and give that grid function back
def add_noise(gf, level, fes, mesh,peaks = 128):
    gf_new = GridFunction(gf.space)
    gf_new.vec.data = gf.vec.Copy()
    level = norm(gf,mesh) * level
    np_random = 2*np.random.rand(peaks,peaks)-1
    scaling = 1/norm(to_gridfunc(np_random, gf.space,mesh),mesh)
    noise = to_gridfunc(scaling*level*np_random,gf.space, mesh)
    gf_new.vec.data += noise.vec.data
    return gf_new, noise


#plot the error plots of u and q and save them
def plot_error(tstart,tend,tsteps, noise_level,error_u,error_q):
    T_u = np.linspace(tstart,int(len(error_u)*0.001),len(error_u))
    T_q = np.linspace(tstart,int(len(error_q)*0.001),len(error_q))

    #general font prep
    font_path = "cmunrm.ttf"
    prop = fm.FontProperties(fname=font_path, size=15)
    prop13 = fm.FontProperties(fname=font_path, size=13)
    prop11 = fm.FontProperties(fname=font_path, size=11)

    font_path = "cmunrm.ttf"
    prop = fm.FontProperties(fname=font_path, size=15)
    prop13 = fm.FontProperties(fname=font_path, size=13)
    prop11 = fm.FontProperties(fname=font_path, size=11)

    np.save("error_u "+str(noise_level)+".npy", np.array(error_u))
    np.save("error_q"+str(noise_level)+".npy", np.array(error_q))

    #plot u error
    if noise_level == 0:
        plt.plot(T_u, error_u)
    else: 
        plt.plot(T_u, error_u,linewidth=0.25)
    plt.yscale('log')
    plt.xlabel("time", fontproperties = prop13)
    plt.ylabel("error", fontproperties = prop13)
    if noise_level == 0: 
        #plt.title("State error (without noise)", fontproperties = prop)
        plt.savefig("State error (without noise).jpg", dpi=300)
    else: 
        #plt.title("State error (with noise)", fontproperties = prop)
        plt.savefig("State error (with noise)"+str(noise_level)+".jpg", dpi=300)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(prop11)
    plt.show()

    #plot q error
    if noise_level == 0:
        plt.plot(T_q, error_q)
    else: plt.plot(T_q,error_q,linewidth=0.25)
    plt.yscale('log')
    plt.xlabel("time", fontproperties = prop13)
    plt.ylabel("error", fontproperties = prop13)
    if noise_level == 0: 
        #plt.title("Parameter error (without noise)", fontproperties = prop)
        plt.savefig("Parameter error (without noise).jpg", dpi=300)
    else: 
        #plt.title("Parameter error (with noise)", fontproperties = prop)
        plt.savefig("Parameter error (with noise)"+str(noise_level)+".jpg", dpi=300)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(prop11)
    plt.show()
    #plot u and q error
    plt.plot(T_u, error_u, color='blue', label='State error')
    plt.plot(T_q, error_q, color='red', label='Parameter error')
    plt.yscale('log')
    plt.xlabel("time", fontproperties = prop13)
    plt.ylabel("error", fontproperties = prop13)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(prop11)
    if noise_level == 0: 
        #plt.title("Parameter and state error (without noise)", fontproperties = prop)
        plt.savefig("Parameter and State error (without noise).jpg", dpi=300)
    else: 
        #plt.title("Parameter and state error (with noise)", fontproperties = prop)
        plt.savefig("Parameter and State error (with noise)"+str(noise_level)+".jpg", dpi=300)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(prop11)
    plt.legend(prop=prop11)
    plt.show()
    np.save("error_u_"+str(noise_level)+".npy", np.array(error_u))
    np.save("error_q_"+str(noise_level)+".npy", np.array(error_q))

#draw the exact and computed u and q
def draw_gridfuncs (tsteps,u_exact,q_exact,gfu,gfq,mesh):
    Draw(gfu, settings={"Objects": {"Wireframe": False}})
    Draw(gfq, settings={"Objects": {"Wireframe": False}})
    try:
        Draw(u_exact,mesh, settings={"Objects": {"Wireframe": False}})
    except:
        Draw(u_exact,mesh, settings={"Objects": {"Wireframe": False}})
    Draw(q_exact,mesh, settings={"Objects": {"Wireframe": False}})
    
#save the mesh, the computed gridfunctions
def save (mesh,noise_level,saved_gfu,saved_gfq):
    mesh.ngmesh.Save("mesh.vol")
    stacked_u = np.stack(saved_gfu, axis=0)  # shape (n_timesteps, n_dofs)
    np.save("gfu_over_time_"+str(noise_level)+".npy", stacked_u)
    stacked_q = np.stack(saved_gfq, axis=0)  # shape (n_timesteps, n_dofs)
    np.save("gfq_over_time_"+str(noise_level)+".npy", stacked_q)