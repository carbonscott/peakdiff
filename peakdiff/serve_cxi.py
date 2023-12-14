#!/usr/bin/env python

import yaml
import logging
import argparse

from bokeh.server.server                 import Server
from bokeh.application                   import Application
from bokeh.application.handlers.function import FunctionHandler

from peakdiff.data import CXIPeakDiff, CXIKeyConfig, CXIConfig, CXIPeakDiffConfig
from peakdiff.viewer import CXIPeakDiffViewer

def create_peakdiff_viewer(config_dict):
    cxi_key_config_0 = CXIKeyConfig(**config_dict['cxi_config_0']['key'])
    cxi_key_config_1 = CXIKeyConfig(**config_dict['cxi_config_1']['key'])

    config_dict['cxi_config_0']['key'] = cxi_key_config_0
    config_dict['cxi_config_1']['key'] = cxi_key_config_1

    cxi_config_0 = CXIConfig(**config_dict['cxi_config_0'])
    cxi_config_1 = CXIConfig(**config_dict['cxi_config_1'])

    dir_output     = config_dict['dir_output']
    uses_cxi_0_img = config_dict['uses_cxi_0_img']
    cxi_peakdiff_config = CXIPeakDiffConfig(cxi_config_0   = cxi_config_0,
                                            cxi_config_1   = cxi_config_1,
                                            dir_output     = dir_output,
                                            uses_cxi_0_img = uses_cxi_0_img,)

    cxi_peakdiff = CXIPeakDiff(cxi_peakdiff_config)
    viewer = CXIPeakDiffViewer(cxi_peakdiff = cxi_peakdiff)

    return viewer


def create_document(doc, path_yaml):
    with open(path_yaml, 'r') as file:
        config_dict = yaml.safe_load(file)

    viewer = create_peakdiff_viewer(config_dict)
    doc.add_root(viewer.final_layout)


def run_bokeh_server(path_yaml, port, websocket_origin):
    logging.basicConfig(level=logging.INFO)
    bokeh_logger = logging.getLogger('bokeh')
    bokeh_logger.setLevel(logging.INFO)

    # Create a Bokeh Application with the specified yaml
    bokeh_app = Application(FunctionHandler(lambda doc: create_document(doc, path_yaml)))

    # Define server settings
    server_settings = {
        'port': port,
        'allow_websocket_origin': [websocket_origin]
    }

    # Create and start the Bokeh server
    server = Server({'/': bokeh_app}, **server_settings)
    server.start()

    server.io_loop.start()


def main():
    parser = argparse.ArgumentParser(description='Run the Bokeh PeakDiff Visualizer.')
    parser.add_argument('--yaml', help='Path to the YAML configuration file')
    parser.add_argument('--port', help='Port to serve the application on', type=int, default=8080)
    parser.add_argument('--websocket-origin', help='WebSocket origin', default='localhost:8080')
    args = parser.parse_args()

    print(f"Starting PeakDiff Visualizer (CXI version) on http://localhost:{args.port}...")

    run_bokeh_server(args.yaml, args.port, args.websocket_origin)


if __name__ == '__main__':
    main()
