import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from matplotlib.patches import Rectangle
import argparse


def plot_heatmap_results(path_to_results_summary_csv: str, save_dir: str):
    """
    Generate a heatmap based on the results summary .csv file

    :param path_to_results_summary_csv: path to the results summary .csv file
    :param save_dir: directory to save the plots
    """
    path_to_results_summary_csv = pathlib.Path(path_to_results_summary_csv)
    save_dir = pathlib.Path(save_dir) if save_dir is not None else path_to_results_summary_csv.parents[0]
    fig, ax = plt.subplots(figsize=(12, 6))

    results_overview = pd.read_csv(path_to_results_summary_csv)
    results_overview.set_index("phenotype", inplace=True)
    models = [mod for mod in results_overview.columns if mod != 'phenotype']

    if 'nested' in path_to_results_summary_csv.parts[-1]:
        types = ['mean', 'std']
    else:
        types = ['mean']
    plot_data_full = pd.DataFrame(columns=[mod + '_' + type for mod in models for type in types] + ['phenotype'])
    plot_data_full['phenotype'] = results_overview.index
    plot_data_full.set_index("phenotype", inplace=True)
    for row in results_overview.iterrows():
        for model in models:
            result = row[1][model]
            if 'std' in types:
                plot_data_full.at[row[0], model + '_std'] = float(result.split('+-')[1])
                plot_data_full.at[row[0], model + '_mean'] = float(result.split('+-')[0])
            else:
                plot_data_full.at[row[0], model + '_mean'] = float(result)
    plot_data_mean = plot_data_full.filter(regex='mean').astype(float)
    row_max = plot_data_mean.idxmax(axis=1)
    sns.heatmap(data=plot_data_mean, cmap="Spectral", cbar_kws={"shrink": .75},
                annot=results_overview, fmt='', linewidths=1.5, linecolor='white', cbar=True, annot_kws={"size": 12})
    ax.set_xticklabels(results_overview.columns, rotation=0)
    ax.set_yticklabels(plot_data_full.index, rotation=0)
    ax.tick_params(top=False,
                   bottom=False,
                   left=False,
                   right=False,
                   labelleft=True,
                   labelbottom=True)
    for row, index in enumerate(plot_data_mean.index):
        position = results_overview.columns.get_loc(row_max[index].split('_')[0])
        ax.add_patch(Rectangle((position, row), 1, 1, fill=False, edgecolor='0', lw=1.5))

    fig.tight_layout()
    plt.savefig(save_dir.joinpath('heatmap_' + path_to_results_summary_csv.parts[-1].split('.')[0] + '.pdf'),
                bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    """
    Run to generate the specified plot based on the results summary .csv file
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-rsp", "--results_summary_path", type=str,
                        help="Provide the full path to the results summary .csv file that you want to plot. "
                             "If not generated yet, run postprocess.results_analysis first.")
    parser.add_argument("-sd", "--save_dir", type=str, default=None,
                        help="Define save directory for the plots. Default is the same as results summary.")
    parser.add_argument("-plot", "--plot", type=str, default='heatmap',
                        help="select plot type: 'heatmap' | ")
    args = vars(parser.parse_args())
    results_summary_path = args['results_summary_path']
    save_dir = args['save_dir']
    if args['plot'] == 'heatmap':
        plot_heatmap_results(path_to_results_summary_csv=results_summary_path, save_dir=save_dir)
    else:
        print('Specified plot type ' + args['plot'] + ' not available. Please check help or source code.')
