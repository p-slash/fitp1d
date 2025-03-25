import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from getdist import plots as getdist_plots


def plotEllipseMinimizer(
        mini, key1, key2, param_labels, color,
        ax=None, label=None, box=False, truth={}, prior={},
        **kwargs
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

    ax.plot(mean_x, mean_y, '+', c=color)
    if key1 in truth:
        ax.axvline(truth[key1], ls=':', c='r')
    if key2 in truth:
        ax.axhline(truth[key2], ls=':', c='r')
    if key1 in truth and key2 in truth:
        ax.plot(truth[key1], truth[key2], 'rx')

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

        if key1 in prior and key2 in prior:
            px, py = prior[key1], prior[key2]
            ax.add_patch(Ellipse(
                (mean_x, mean_y), width=px * 2 * n, height=py * 2 * n, ls=ls,
                fill=False, ec='k', lw=2, **kwargs
            ))
        elif key1 in prior:
            px = prior[key1]
            ax.axvspan(mean_x - n * px, mean_x + n * px,
                       alpha=0.5 * alpha / n, fc='k')
        elif key2 in prior:
            py = prior[key2]
            ax.axhspan(mean_y - n * py, mean_y + n * py,
                       alpha=0.5 * alpha / n, fc='k')

    ax.set_xlabel(rf"${param_labels[key1]}$")
    ax.set_ylabel(rf"${param_labels[key2]}$")
    return ax


def plotAllEllipses(likeli, ofname=None):
    free_params = likeli.free_params
    mini = likeli._mini
    model = likeli

    nplots = len(free_params)
    nplots *= nplots - 1
    nplots //= 2
    ncols = round(np.sqrt(nplots))
    nrows = int(np.ceil(nplots / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 5 * nrows),
        gridspec_kw={'hspace': 0.2, 'wspace': 0.3})

    j = 0
    for i, key1 in enumerate(free_params[:-1]):
        for key2 in free_params[i + 1:]:
            plotEllipseMinimizer(
                mini, key1, key2, model.param_labels, 'tab:blue',
                ax=axs[j // ncols, j % ncols],
                alpha=0.6, box=True, truth=model.initial, prior=model.prior
            )
            j += 1

    if ofname:
        plt.savefig(ofname, dpi=200, bbox_inches='tight')
    plt.close()


def plotCornerSamples(
        list_samples, vars2plot=None, contour_colors=None, ofname=None,
        show=True, truth={}
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
        legend_loc='upper right',
        markers=truth, marker_args={'lw': 1}
    )

    if ofname:
        plt.savefig(ofname, dpi=150, bbox_inches='tight')

    if show:
        plt.show()


def plotFitNData(likeli, plot_ratio=True):
    data = likeli._data
    k = data['kc']
    fit = likeli._mini.values.to_dict()
    model = likeli.p1dmodel.getIntegratedModel(**fit)

    if not plot_ratio:
        plt.errorbar(
            k, data['p_final'] * k / np.pi, data['e_total'] * k / np.pi,
            fmt='o', capsize=2)

        plt.plot(k, model * k / np.pi, 'k-')
        plt.yscale("log")
        plt.ylabel(r"$kP/\pi$")
    else:
        plt.errorbar(
            k, data['p_final'] / model, data['e_total'] / model,
            fmt='o', capsize=2)
        plt.ylabel(r"$P_\mathrm{data}/P_\mathrm{bestfit}$")

    plt.xscale("log")
    plt.xlabel(r"$k$ [s km$^{-1}$]")

    return plt.gca()
