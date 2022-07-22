import argparse

from . import results_analysis

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
        results_analysis.plot_heatmap_results(path_to_results_summary_csv=results_summary_path, save_dir=save_dir)
    else:
        print('Specified plot type ' + args['plot'] + ' not available. Please check help or source code.')
