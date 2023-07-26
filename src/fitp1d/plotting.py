import numpy as np
import matplotlib.pyplot as plt

from getdist import plots as getdist_plots


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


def plotFitNData(likeli, show=True):
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
    plt.grid(True)
    plt.ylabel(r"$kP/\pi$", fontsize=16)
    plt.xlabel(r"$k$ [s km$^{-1}$]", fontsize=16)
    if show:
        plt.show()
