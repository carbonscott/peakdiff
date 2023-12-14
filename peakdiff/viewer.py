#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bokeh.models         import ColumnDataSource, Circle, Div, CustomJS, Span, HoverTool, PolyDrawTool, DataTable, TableColumn, Button
from bokeh.plotting       import figure, show
from bokeh.layouts        import gridplot, row, column
from bokeh.io             import curdoc
from bokeh.models.mappers import LogColorMapper


class CXIPeakDiffViewer:

    def __init__(self, cxi_peakdiff, fig_height = 600, fig_width = 600, num_cpus = 20):
        self.fig_height   = fig_height
        self.fig_width    = fig_width
        self.cxi_peakdiff = cxi_peakdiff

        scatter_plot_data_source      = self.cxi_peakdiff.build_bokeh_data_source(path_metrics = None, num_cpus = num_cpus)
        self.scatter_plot_data_source = ColumnDataSource(scatter_plot_data_source)

        self.init_scatterplot_panel()
        self.init_image_panel()
        self.init_events_panel()
        self.init_section_title_panel()
        self.init_layout()


    def init_scatterplot_panel(self):
        fig_height   = self.fig_height
        fig_width    = self.fig_width
        title_width  = 30
        title_height = fig_height

        scatter_plot_data_source = self.scatter_plot_data_source

        TOOLS = "tap,box_select,lasso_select,wheel_zoom,pan,reset,help,"

        fig = dict(
            n_peaks = figure(width        =  fig_width,
                             height       =  fig_height,
                             tools        =  TOOLS,
                             title        = "Number of peaks comparison",
                             x_axis_label = 'pyalgo',
                             y_axis_label = 'peaknet',
                             match_aspect = True,
                             active_scroll="wheel_zoom"),
            m_rates = figure(width        =  fig_width,
                             height       =  fig_height,
                             tools        =  TOOLS,
                             title        = "Match rate comparison",
                             x_axis_label = 'pyalgo',
                             y_axis_label = 'peaknet',
                             match_aspect = True,
                             active_scroll="wheel_zoom"),
        )

        scatter_n_peaks = fig['n_peaks'].scatter('n_peaks_x',
                                                 'n_peaks_y',
                                                 source                  = scatter_plot_data_source,
                                                 size                    = 10,
                                                 fill_color              = "blue",
                                                 line_color              =  None,
                                                 fill_alpha              = 0.5,
                                                 nonselection_fill_alpha = 0.005,
                                                 nonselection_fill_color = "blue")
        scatter_m_rates = fig['m_rates'].scatter('m_rates_x',
                                                 'm_rates_y',
                                                 source                  = scatter_plot_data_source,
                                                 size                    = 10,
                                                 fill_color              = "red",
                                                 line_color              =  None,
                                                 fill_alpha              = 0.5,
                                                 nonselection_fill_alpha = 0.005,
                                                 nonselection_fill_color = "red")

        self.fig = fig

        scatter_plot_data_source.selected.on_change('indices', self.update_events_panel)


    def update_events_panel(self, attr, old, new):
        # [NOTE] Indices is a concept in bokeh data source.  Event is a concept in CXI.
        selected_indices = self.scatter_plot_data_source.selected.indices
        selected_events  = [ self.scatter_plot_data_source.data['events'][item] for item in selected_indices ]
        self.selected_events_data_source.data = {'events': selected_events}

        # Clear the image panel if no events are selected
        if not selected_events:
            self.clear_image_panel()


    def clear_image_panel(self):
        self.img_data_source.data = {'x': [], 'y': [], 'dw': [], 'dh': [], 'f': []}

        self.rect_peak_data_source_0.data = dict(
            x      = [],
            y      = [],
            width  = [],
            height = [],
        )
        self.rect_peak_data_source_1.data = dict(
            x      = [],
            y      = [],
            width  = [],
            height = [],
        )


    def update_image_panel(self, img, peaks_0, peaks_1):
        # Create ColumnDataSource
        H, W = img.shape

        vmin = img.mean()
        vmax = img.mean() + 6 * img.std()
        color_mapper = LogColorMapper(palette="Viridis256", low=vmin, high=vmax)

        self.image_glyph.glyph.color_mapper = color_mapper

        self.img_data_source.data = data=dict(
            x  = [0],
            y  = [0],
            dw = [W],
            dh = [H],
            f  = [img],
        )

        offset = 3
        y, x = peaks_0
        self.rect_peak_data_source_0.data = data=dict(
            x      = x,
            y      = y,
            width  = [2*offset] * len(x),
            height = [2*offset] * len(y),
        )
        y, x = peaks_1
        self.rect_peak_data_source_1.data = data=dict(
            x      = x,
            y      = y,
            width  = [2*offset] * len(x),
            height = [2*offset] * len(y),
        )


    def init_image_panel(self):
        fig_height   = self.fig_height
        fig_width    = self.fig_width * 2

        fig = figure(width        =  int(fig_width),
                     height       =  int(fig_height),
                     y_range      =  (0, fig_height),
                     x_range      =  (0, fig_width),
                     match_aspect = True,
                     active_scroll="wheel_zoom")

        self.img_data_source = ColumnDataSource(data=dict(
            x  = [],
            y  = [],
            dw = [],
            dh = [],
            f  = [],
        ))
        color_mapper = LogColorMapper(palette="Viridis256")
        self.image_glyph = fig.image(source = self.img_data_source, image = 'f', x = 'x', y = 'y', dw = 'dw', dh = 'dh', color_mapper=color_mapper)

        self.rect_peak_data_source_0 = ColumnDataSource(data=dict(
            x      = [],
            y      = [],
            width  = [],
            height = [],
        ))
        fig.rect(source     =  self.rect_peak_data_source_0,
                 x          =  'x',
                 y          =  'y',
                 width      =  'width',
                 height     =  'height',
                 line_width = 1.0,
                 line_color = 'yellow',
                 fill_color = None)

        self.rect_peak_data_source_1 = ColumnDataSource(data=dict(
            x      = [],
            y      = [],
            width  = [],
            height = [],
        ))
        fig.rect(source     =  self.rect_peak_data_source_1,
                 x          =  'x',
                 y          =  'y',
                 width      =  'width',
                 height     =  'height',
                 line_width = 1.0,
                 line_color = 'cyan',
                 fill_color = None)

        # Add hover tool with the callback
        hover_tool = HoverTool(
            tooltips   = [('x', '$x{%d}'), ('y', '$y{%d}'), ('v', '@f{0.2f}')],
            formatters = {
                '$x': 'printf',
                '$y': 'printf',
                '@v': 'printf'
            },
            renderers = [self.image_glyph])
        fig.add_tools(hover_tool)

        scatter_plot_data_source = self.scatter_plot_data_source
        cxi_peakdiff             = self.cxi_peakdiff

        def load_selected(attr, old, new):
            if len(new) == 1:
                selected_index = new[0]
                event = self.scatter_plot_data_source.data['events'][selected_index]

                # Uses cxi_0 images???
                img = cxi_peakdiff.cxi_manager_0.get_img_by_event(event) \
                      if cxi_peakdiff.uses_cxi_0_img else                \
                      cxi_peakdiff.cxi_manager_1.get_img_by_event(event)

                peaks_0 = cxi_peakdiff.cxi_manager_0.get_peaks_by_event(event)
                peaks_1 = cxi_peakdiff.cxi_manager_1.get_peaks_by_event(event)

                self.update_image_panel(img, peaks_0, peaks_1)

        scatter_plot_data_source.selected.on_change('indices', load_selected)

        self.selected_event_div = fig


    def init_events_panel(self):
        fig_height   = self.fig_height
        fig_width    = self.fig_width
        title_width  = 30
        title_height = fig_height

        scatter_plot_data_source = self.scatter_plot_data_source

        # Create a data source for the DataTable
        self.selected_events_data_source = ColumnDataSource(data={'events': []})

        # Define columns for the DataTable
        columns = [TableColumn(field="events", title="Selected Indices")]

        # Create the DataTable
        self.selected_events_table = DataTable(source=self.selected_events_data_source,
                                                columns=columns,
                                                width=2 * fig_width,
                                                height=title_height,
                                                selectable=True,
                                                header_row=False)

        self.selected_events_table.source.selected.on_change('indices', self.on_table_select)


    def on_table_select(self, attr, old, new):
        if new:
            event = self.selected_events_data_source.data['events'][new[0]]
            ## selected_index = self.selected_indices_data_source.data['indices'][new[0]]
            ## event = self.scatter_plot_data_source.data['events'][selected_index]
            self.load_selected_event_by_index(event)
        else:
            self.clear_image_panel()


    def load_selected_event_by_index(self, event):
        cxi_peakdiff = self.cxi_peakdiff

        # Uses cxi_0 images???
        img = cxi_peakdiff.cxi_manager_0.get_img_by_event(event) \
              if cxi_peakdiff.uses_cxi_0_img else                \
              cxi_peakdiff.cxi_manager_1.get_img_by_event(event)

        peaks_0 = cxi_peakdiff.cxi_manager_0.get_peaks_by_event(event)
        peaks_1 = cxi_peakdiff.cxi_manager_1.get_peaks_by_event(event)

        self.update_image_panel(img, peaks_0, peaks_1)


    def init_section_title_panel(self):
        fig_height   = self.fig_height
        fig_width    = self.fig_width
        title_width  = 30
        title_height = fig_height

        scatter_plot_data_source = self.scatter_plot_data_source

        # Section title...
        section_div = {}

        div_text = """
        <div style="
            background-color: green;
            color           : white;
            padding         : 5px;
            writing-mode    : vertical-lr;
            transform       : rotate(180deg);
            height          : 100%;
            font-weight     : bold;
            font-size       : 18px;
            text-align      : center;
            white-space     : nowrap;
        ">
            Scatter Plot
        </div>
        """
        section_div['scatter_plot'] = Div(text = div_text, width = title_width, height = title_height)
        section_div['scatter_plot'].margin = (0, 5, 0, 0)

        div_text = """
        <div style="
            background-color: gray;
            color           : white;
            padding         : 5px;
            writing-mode    : vertical-lr;
            transform       : rotate(180deg);
            height          : 100%;
            font-weight     : bold;
            font-size       : 18px;
            text-align      : center;
            white-space     : nowrap;
        ">
            Selected event
        </div>
        """
        section_div['selected_event'] = Div(text = div_text, width = title_width, height = title_height)
        section_div['selected_event'].margin = (0, 5, 0, 0)

        div_text = """
        <div style="
            background-color: blue;
            color           : white;
            padding         : 5px;
            writing-mode    : vertical-lr;
            transform       : rotate(180deg);
            height          : 100%;
            font-weight     : bold;
            font-size       : 18px;
            text-align      : center;
            white-space     : nowrap;
        ">
            Selected Indices
        </div>
        """
        section_div['selected_events'] = Div(text = div_text, width = title_width, height = title_height)
        section_div['selected_events'].margin = (0, 5, 0, 0)

        self.section_div = section_div


    def init_layout(self):
        fig                  = self.fig
        selected_event_div   = self.selected_event_div
        selected_events_table = self.selected_events_table
        section_div          = self.section_div

        layout = {}
        layout['scatter_plot'    ] = row(section_div['scatter_plot']    , gridplot([[fig['n_peaks'], fig['m_rates']]], toolbar_location = 'right'))
        layout['selected_event'  ] = row(section_div['selected_event']  , selected_event_div)
        layout['selected_events'] = row(section_div['selected_events'], selected_events_table)

        final_layout = column(layout['scatter_plot'], layout['selected_event'], layout['selected_events'])

        self.final_layout = final_layout


    def run(self):
        curdoc().add_root(self.final_layout)
