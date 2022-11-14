from src.Experiments import plots, exp_config

config = exp_config.get_exp_config()
config["save_path"] = ""
config["layerwise_training"] = True

plots.produce_usl_lineval_plots(config, load_path="data.csv")
