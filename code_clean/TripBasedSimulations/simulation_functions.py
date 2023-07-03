import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import pandas as pd
from scipy.optimize import root, root_scalar
plt.rcParams['text.usetex'] = True
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from turtle import color

SEED = 1000
# THETA = 1000000
# BETA = 10
kj = 2000
v0 = 1.0
α = 560
ω = 40
ε = np.finfo(float).eps
V = np.vectorize(lambda k: np.maximum(v0*(1 - k/kj),1/1000))
μ = lambda k: 1./V(k)
def D(u):
    if u > α/ω:
        return 0
    return α - ω*u
D = np.vectorize(D)
def Dinv(A):
    if A > α:
        return 0
    return (α-A)/ω
Dinv = np.vectorize(Dinv)
f = lambda k: k*V(k)
rng = np.random.default_rng(SEED)
normal_σ = 1.
normal_μ = -0.5
rnd_dev = lambda: rng.lognormal(normal_μ, normal_σ)



#to plot
keq_0 = root_scalar(lambda k: D(μ(k)) - f(k), bracket = [1500, 1950]).root
keq_1 = root_scalar(lambda k: D(μ(k)) - f(k), bracket = [1000, 1500]).root
keq_2 = root_scalar(lambda k: D(μ(k)) - f(k), bracket = [250, 1000]).root

#to add arrows to plot
def add_arrow_to_line2D(
    axes, line, arrow_locs=[0.005, 0.01],
    arrowstyle='-|>', arrowsize=1, transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes: 
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if not isinstance(line, mlines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
        "arrowstyle": arrowstyle,
        "mutation_scale": 10 * arrowsize,
    }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows

def plot_cumm_curve(sims, model, save = None):
    fig, ax = plt.subplots(figsize=(8,6))
    color = ['red', 'blue', 'green', 'orange']
    for i, sim in enumerate(sims):
        if model == 'k':
            leg = "("+str(sim.start_density)+")"
        elif model == 'kA':
            leg = "("+str(sim.start_density)+"," + str(sim.start_inflow)+")"
        elif model == 'ku':
            leg = "("+str(sim.start_density)+"," + str(sim.start_u)+")"
        elif model == 'kAu':
            leg = "("+str(sim.start_density)+"," + str(sim.start_inflow) + "," +  str(sim.start_u)+")"
        cumm = sim.get_cumm_curve()
        time = datetime.datetime.now()
        cumm['time'] = cumm.apply(lambda row: time + timedelta(seconds=row['t']), axis=1)
        cumm['time'] = pd.to_datetime(cumm['time'])
        cumm = cumm.set_index('time')
        cumm['arr'] = np.zeros_like(cumm.t)
        cumm.arr[cumm.type == 1] = 1
        cumm['dep'] = np.zeros_like(cumm.t)
        cumm.dep[cumm.type == 0] = 1
        cumm['cumm_arr'] = cumm.arr.cumsum()
        cumm['cumm_dep'] = cumm.dep.cumsum()
        cumm.head()
        # ax.plot(cumm.n.rolling('5s').mean(), cumm.arr.rolling('5s').sum()/5, label = 'Arrivals: '+str(i+1), color = color[i])
        # ax.plot(cumm.n.rolling('5s').mean(), cumm.dep.rolling('5s').sum()/5, label = 'Exits: '+str(i+1), color = color[i], ls = '--')
        ax.plot(cumm.n.rolling('5s').mean(), cumm.arr.rolling('5s').sum()/5, label = 'Arrivals: ' + leg, color = color[i])
        ax.plot(cumm.n.rolling('5s').mean(), cumm.dep.rolling('5s').sum()/5, label = 'Exits: ' + leg, color = color[i], ls = '--')
    
    ks = np.linspace(0,kj,500)
    ax.plot(ks,f(ks), alpha=0.2, color = 'k')
    ax.plot(ks,D(μ(ks)),alpha=0.2, color = 'k')
    ax.set(xlabel = r'$k$', ylabel = r'$q$')
    ax.legend()
    if save:
        fig.savefig(save)

    
def plot_Tn_curve(sims, model, save = None):
    fig, ax = plt.subplots(figsize=(8,6))
    color = ['red', 'blue', 'green', 'orange']
    for i, sim in enumerate(sims):
        if model == 'k':
            leg = 'k: '+  "("+str(sim.start_density)+")"
        elif model == 'kA':
            leg =  'k,A: '+"("+str(sim.start_density)+"," + str(sim.start_inflow)+")"
        elif model == 'ku':
            leg = 'k,u: '+"("+str(sim.start_density)+"," + str(sim.start_u)+")"
        elif model == 'kAu':
            leg = 'k,A,u: ' + "("+str(sim.start_density)+"," + str(sim.start_inflow) + "," +  str(sim.start_u)+")"

        cumm = sim.get_cumm_curve()
        time = datetime.datetime.now()
        cumm['time'] = cumm.apply(lambda row: time + timedelta(seconds=row['t']), axis=1)
        cumm['time'] = pd.to_datetime(cumm['time'])
        cumm = cumm.set_index('time')
        # ax.plot(cumm.t.rolling('5s').mean(), cumm.n.rolling('5s').mean(), label = "Run "+str(i+1), color = color[i])
        ax.plot(cumm.t.rolling('5s').mean(), cumm.n.rolling('5s').mean(), label = leg, color = color[i])
    
    ax.hlines(keq_0, 0, 180, ls = '--', color = 'k', alpha = 0.2)
    ax.hlines(keq_1, 0, 180, ls = '--', color = 'k', alpha = 0.2)
    ax.hlines(keq_2, 0, 180, ls = '--', color = 'k', alpha = 0.2)
    y_bounds = ax.get_ylim()
    ax.annotate(text=r'$x$', xy =(1.02, ((keq_0-y_bounds[0])/(y_bounds[1]-y_bounds[0]))), xycoords='axes fraction', verticalalignment='top', horizontalalignment='right' , rotation = 0)
    ax.annotate(text=r'$y$', xy =(1.02, ((keq_1-y_bounds[0])/(y_bounds[1]-y_bounds[0]))), xycoords='axes fraction', verticalalignment='top', horizontalalignment='right' , rotation = 0)
    ax.annotate(text=r'$z$', xy =(1.02, ((keq_2-y_bounds[0])/(y_bounds[1]-y_bounds[0]))), xycoords='axes fraction', verticalalignment='top', horizontalalignment='right' , rotation = 0)
    # ax.annotate(xy=(0, keq_0), s = "x")
    # ax.annotate(xy=(0, keq_1), s = "y")
    # ax.annotate(xy=(0, keq_2), s = "z")

    ax.set_xlim(0, 100)
    ax.set(xlabel = r'$t$', ylabel = r'$k$')
    ax.legend()
    if save:
        fig.savefig(save)