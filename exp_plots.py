from ColabExport import exp_config, plots

config = exp_config.get_exp_config()

plots.produce_usl_lineval_plots(config, load_path="data.csv")
