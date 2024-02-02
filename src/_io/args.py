import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')

