# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
plt.rcParams['text.usetex'] = True
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from turtle import color

# define functions
SEED = 1000
# THETA = 1000000
# BETA = 10
kj = 2000
v0 = 1.0
α = 560
ω = 40
ε = np.finfo(float).eps
v = np.vectorize(lambda k: np.maximum(v0*(1 - k/kj),1/1000))
μ = lambda k: 1./v(k)
def D(u):
    if u > α/ω:
        return 0
    return α - ω*u
D = np.vectorize(D)
def Dinv(A):
    if A > α:
        return 0
    return (α-A)/ω
d = np.vectorize(Dinv)
Z = lambda k: D(μ(k))
f = lambda k: k*v(k)


# add arrow to lines
def add_arrow_to_line2D(
    axes, line, arrow_locs=[0.5],
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