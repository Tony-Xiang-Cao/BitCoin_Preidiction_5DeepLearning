import logging
import os
import json
import argparse

def set_up_app():
    try:
        set_up_logging()
        args = set_up_parsing()
        configs = load_config(args)
        
        return configs
    except:
        logging.info("Something went wrong setting up the app. Exiting.")
        return    
    
def set_up_logging():
    # Setting informational log such that it prints
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Logging set up")

def set_up_parsing():
    # Parsing the argument which should be the configuration file
    arg_parser = argparse.ArgumentParser(description="BTC Price Predictor")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()
    
    return args

def load_config(args):
    # Trying to load the config variables, otherwise exiting program
    try:  
        logging.info(f"Trying to load configuration variables from {args.config}")
        configs = json.load(open(args.config, 'r'))
        logging.info(f"Successfully loaded configuration file!")
    except:
        logging.info("Either no config exists or there was an error reading the config file. Exiting")
        return
    
    # Creating directories to save the model
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    
    return configs
