#!/usr/bin/env python

import yaml
import logging
import argparse

from bokeh.server.server                 import Server
from bokeh.application                   import Application
from bokeh.application.handlers.function import FunctionHandler

from peakdiff.crystfel_stream.data import StreamPeakDiff, StreamConfig, StreamPeakDiffConfig
from peakdiff.crystfel_stream.viewer import StreamPeakDiffViewer

def create_peakdiff_viewer(config_dict):
    stream_config          = StreamConfig(**config_dict['stream_config'])
    ignores_cache          = config_dict['ignores_cache']
    dir_output             = config_dict['dir_output'   ]
    num_cpus               = config_dict['num_cpus']
    stream_peakdiff_config = StreamPeakDiffConfig(stream_config = stream_config, ignores_cache = ignores_cache, dir_output = dir_output)
    stream_peakdiff        = StreamPeakDiff(stream_peakdiff_config)
    viewer                 = StreamPeakDiffViewer(stream_peakdiff = stream_peakdiff, num_cpus = num_cpus)

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

    try:
        # Define server settings
        server_settings = {
            'port': port,
            'allow_websocket_origin': [websocket_origin]
        }

        # Create and start the Bokeh server
        server = Server({'/': bokeh_app}, **server_settings)
        server.start()

        server.io_loop.start()

    except KeyboardInterrupt:
        print("Shutting down...")
        server.stop()
        server.io_loop.stop()


def main():
    parser = argparse.ArgumentParser(description='Run the Bokeh PeakDiff Visualizer.')
    parser.add_argument('--yaml', help='Path to the YAML configuration file')
    parser.add_argument('--port', help='Port to serve the application on', type=int, default=8080)
    parser.add_argument('--websocket-origin', help='WebSocket origin', default='localhost:8080')
    args = parser.parse_args()

    print(f"Starting PeakDiff Visualizer (CrystFEL Stream version) on http://localhost:{args.port}...")

    run_bokeh_server(args.yaml, args.port, args.websocket_origin)


if __name__ == '__main__':
    main()
