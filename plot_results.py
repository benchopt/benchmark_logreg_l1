import re
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl


usetex = mpl.checkdep_usetex(True)
params = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Roman"],
    "text.usetex": usetex,
}
mpl.rcParams.update(params)


# SAVEFIG = False
SAVEFIG = True
figname = "logreg_l1"

# RUN `benchopt run . --config config_medium.yml`, then replace BENCH_NAME
# by the name of the produced results csv file.

BENCH_NAME = "outputs/logreg-l1_neurips.csv"

FLOATING_PRECISION = 1e-8
MIN_XLIM = 1e-3
MARKERS = list(plt.Line2D.markers.keys())[:-4]


SOLVERS = {
    'Blitz': "blitz",
    'cd[newton_step=False]': 'coordinate descent',
    'cd[newton_step=True]': 'coordinate descent (Newton)',
    'Celer': 'celer',
    'copt[accelerated=False,line_search=False,solver=pgd]': 'copt (PGD)',
    'copt[accelerated=False,line_search=True,solver=pgd]': (
        'copt (PGD line search)'
    ),
    'copt[accelerated=True,line_search=False,solver=pgd]': 'copt (FISTA)',
    'copt[accelerated=True,line_search=True,solver=pgd]': (
        'copt (FISTA line search)'
    ),
    'Cyanure': 'cyanure',
    'cuml[qn]': 'cuML[qn]',
    'Lightning': 'lightning',
    'sklearn[liblinear]': 'liblinear',
    'copt[accelerated=False,line_search=False,solver=svrg]': 'copt (SVRG)',
    'copt[accelerated=False,line_search=False,solver=saga]': 'copt (SAGA)',
    'snapml[gpu=False]': 'snapML[cpu]',
    'snapml[gpu=True]': 'snapML[gpu]',
}

all_solvers = SOLVERS.keys()

DICT_XLIM = {
    "libsvm[dataset=rcv1.binary]": 1e-2,
    "libsvm[dataset=news20.binary]": 1e-1,
    "libsvm[dataset=colon-cancer]": 1e-4,
    "libsvm[dataset=gisette]": 1e-2,
}

DICT_TITLE = {
    'Sparse Logistic Regression[fit_intercept=False,reg=0.1]': (
        r'$\lambda = 0.1 \lambda_{\mathrm{max}}$'
    ),
    'Sparse Logistic Regression[fit_intercept=False,reg=0.01]': (
        r'$\lambda = 0.01 \lambda_{\mathrm{max}}$'
    ),
    'Sparse Logistic Regression[fit_intercept=False,reg=0.001]': (
        r'$\lambda = 0.001 \lambda_{\mathrm{max}}$'
    ),
}

DICT_YLABEL = {
    'libsvm[dataset=rcv1.binary]': "rcv1.binary",
    'libsvm[dataset=news20.binary]': "news20.binary",
    'libsvm[dataset=gisette]': "gisette",
    'libsvm[dataset=colon-cancer]': "colon-cancer",
}

DICT_YTICKS = {
    'libsvm[dataset=rcv1.binary]': [1e3, 1, 1e-3, 1e-6],
    'libsvm[dataset=news20.binary]': [1e3, 1, 1e-3, 1e-6],
    'libsvm[dataset=gisette]': [1e3, 1, 1e-3, 1e-6],
    'libsvm[dataset=colon-cancer]': [1e3, 1, 1e-3, 1e-6],
}

DICT_XTICKS = {
    'libsvm[dataset=rcv1.binary]': np.geomspace(1e-2, 1e2, 5),
    'libsvm[dataset=news20.binary]': np.geomspace(1e-1, 1e3, 5),
    'libsvm[dataset=gisette]': np.geomspace(1e-2, 1e2, 5),
    'libsvm[dataset=colon-cancer]': np.geomspace(1e-4, 1e2, 7),
}

CMAP = plt.get_cmap('tab20')
style = {solv: (CMAP(i), MARKERS[i]) for i, solv in enumerate(all_solvers)}


df = pd.read_csv(BENCH_NAME, header=0, index_col=0)


solvers = df["solver_name"].unique()
solvers = np.array(sorted(solvers, key=lambda key: SOLVERS[key].lower()))
datasets = [
    'libsvm[dataset=gisette]',
    'libsvm[dataset=colon-cancer]',
    'libsvm[dataset=rcv1.binary]',
    'libsvm[dataset=news20.binary]',
]

objectives = df["objective_name"].unique()

titlesize = 22
ticksize = 16
labelsize = 20
regex = re.compile(r'\[(.*?)\]')

plt.close('all')
fig1, axarr = plt.subplots(
    len(datasets),
    len(objectives),
    sharex=False,
    sharey='row',
    figsize=[11, 1 + 2 * len(datasets)],
    constrained_layout=True,
    squeeze=False)

for idx_data, dataset in enumerate(datasets):
    df1 = df[df['data_name'] == dataset]
    for idx_obj, objective in enumerate(objectives):
        df2 = df1[df1['objective_name'] == objective]
        ax = axarr[idx_data, idx_obj]
        c_star = np.min(df2["objective_value"]) - FLOATING_PRECISION
        for i, solver_name in enumerate(solvers):
            df3 = df2[df2['solver_name'] == solver_name]
            curve = df3.groupby('stop_val').median()

            q1 = df3.groupby('stop_val')['time'].quantile(.1)
            q9 = df3.groupby('stop_val')['time'].quantile(.9)
            y = curve["objective_value"] - c_star

            linestyle = '-'
            if solver_name in ("snapml[gpu=True]", "cuml[qn]", "cuml[cd]"):
                linestyle = '--'
            ax.loglog(
                curve["time"], y, color=style[solver_name][0],
                marker=style[solver_name][1], markersize=6,
                label=SOLVERS[solver_name], linewidth=2, markevery=3,
                linestyle=linestyle)

        ax.set_xlim([DICT_XLIM.get(dataset, MIN_XLIM), ax.get_xlim()[1]])
        axarr[len(datasets)-1, idx_obj].set_xlabel(
            "Time (s)", fontsize=labelsize
        )
        axarr[0, idx_obj].set_title(
            DICT_TITLE[objective], fontsize=labelsize)

        ax.grid()
        ax.set_xticks(DICT_XTICKS[dataset])
        ax.tick_params(axis='both', which='major', labelsize=ticksize)

    if regex.search(dataset) is not None:
        dataset_label = (regex.sub("", dataset) + '\n' +
                         '\n'.join(regex.search(dataset).group(1).split(',')))
    else:
        dataset_label = dataset
    axarr[idx_data, 0].set_ylabel(DICT_YLABEL[dataset], fontsize=labelsize)
    axarr[idx_data, 0].set_yticks(DICT_YTICKS[dataset])

plt.show(block=False)


fig2, ax2 = plt.subplots(1, 1, figsize=(20, 4))
n_col = 4
if n_col is None:
    n_col = len(axarr[0, 0].lines)

ax = axarr[0, 0]
lines_ordered = list(itertools.chain(
    *[ax.lines[i::n_col] for i in range(n_col)]))
legend = ax2.legend(
    lines_ordered, [line.get_label() for line in lines_ordered], ncol=n_col,
    loc="upper center")
fig2.canvas.draw()
fig2.tight_layout()
width = legend.get_window_extent().width
height = legend.get_window_extent().height
fig2.set_size_inches((width / 80,  max(height / 80, 0.5)))
plt.axis('off')
plt.show(block=False)


if SAVEFIG:
    Path("figures").mkdir(exist_ok=True)
    fig1_name = f"figures/{figname}.pdf"
    fig1.savefig(fig1_name)

    fig2_name = f"figures/{figname}_legend.pdf"
    fig2.savefig(fig2_name)
