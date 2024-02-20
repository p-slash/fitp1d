import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from getdist import plots as getdist_plots


def plotEllipseMinimizer(
        mini, key1, key2, param_labels, color,
        ax=None, label=None, box=False, **kwargs
):
    if ax is None:
        ax = plt.gca()

    std_x = np.sqrt(mini.covariance[(key1, key1)])
    mean_x = mini.values[key1]
    std_y = np.sqrt(mini.covariance[(key2, key2)])
    mean_y = np.mean(mini.values[key2])

    pearson = mini.covariance[(key1, key2)] / (std_x * std_y)

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
        fc=color, ec=color, lw=2, **kwargs
    )

    ax.plot(mean_x, mean_y, 'kx')
    if box:
        textstr = '\n'.join((
            rf'${param_labels[key1]}=$' + f"{mean_x:.3f}",
            rf'${param_labels[key2]}=$' + f"{mean_y:.3f}",
            r"$r=$" + f"{pearson:.2f}"
        ))
        ax.text(
            0.05, 0.95, textstr,
            transform=ax.transAxes, fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.4
    for n, ls in zip([1, 2], ['-', '--']):
        scale_x = n * std_x
        scale_y = n * std_y
        transf = transforms.Affine2D().rotate_deg(45).scale(
            scale_x, scale_y).translate(mean_x, mean_y)
        ell = copy.deepcopy(ellipse)

        ell.set_transform(transf + ax.transData)
        ell.set_linestyle(ls)
        if 'fill' not in kwargs or kwargs['fill']:
            ell.set_alpha(alpha / n)

        if n == 1:
            ell.set_label(label)
        ax.add_patch(ell)

    return ax


def plotCornerSamples(
        list_samples, vars2plot=None, contour_colors=None, ofname=None,
        show=True
):
    gplt = getdist_plots.get_subplot_plotter()
    if not isinstance(list_samples, list):
        x0 = list_samples
    else:
        x0 = list_samples[0]
    if vars2plot is not None:
        s_ = np.sqrt(len(vars2plot) / x0.n)
    else:
        s_ = 1

    gplt.settings.legend_fontsize = 24 * s_ + 8
    gplt.settings.axes_fontsize = 12 * s_ + 8
    gplt.settings.axes_labelsize = 16 * s_ + 8
    gplt.settings.axis_tick_x_rotation = 45

#     gplt.settings.solid_contour_palefactor = 0.9
#     gplt.settings.alpha_filled_add = 0.6

    gplt.triangle_plot(
        list_samples, vars2plot,
        filled=True,
        contour_colors=contour_colors,
        legend_loc='upper right'
    )

    if ofname:
        plt.savefig(ofname, dpi=150, bbox_inches='tight')

    if show:
        plt.show()


def plotFitNData(likeli, ofname=None, show=True):
    data = likeli._data
    k = data['kc']
    plt.errorbar(
        k, data['p_final'] * k / np.pi, data['e_total'] * k / np.pi,
        fmt='o', capsize=2)
    fit = likeli._mini.values.to_dict()
    model = likeli.p1dmodel.getIntegratedModel(**fit)

    plt.plot(k, model * k / np.pi, 'k-')
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, "major")
    plt.grid(True, "minor", linestyle=':', linewidth=1)
    plt.ylabel(r"$kP/\pi$", fontsize=16)
    plt.xlabel(r"$k$ [s km$^{-1}$]", fontsize=16)

    if ofname:
        plt.savefig(ofname, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
