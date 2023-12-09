#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bokeh.models         import ColumnDataSource, Circle, Div, CustomJS, Span, HoverTool, PolyDrawTool
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
        self.init_indices_panel()
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
                             match_aspect = True),
            match_rates = figure(width        =  fig_width,
                                 height       =  fig_height,
                                 tools        =  TOOLS,
                                 title        = "Match rate comparison",
                                 x_axis_label = 'pyalgo',
                                 y_axis_label = 'peaknet',
                                 match_aspect = True),
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
        scatter_match_rates = fig['match_rates'].scatter('match_rates_x',
                                                         'match_rates_y',
                                                         source                  = scatter_plot_data_source,
                                                         size                    = 10,
                                                         fill_color              = "red",
                                                         line_color              =  None,
                                                         fill_alpha              = 0.5,
                                                         nonselection_fill_alpha = 0.005,
                                                         nonselection_fill_color = "red")

        self.fig = fig


    ## def update_image_panel(self, img, peaks_0, peaks_1):
    def update_image_panel(self, img, peaks_1):
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

        ## # Prepare rectangle data for peaks
        ## y, x = peaks_1
        ## self.rect_peak_data_source_0.data = data=dict(
        ##     x      = x,
        ##     y      = y,
        ##     width  = [2*offset] * len(x),
        ##     height = [2*offset] * len(y),
        ## )

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
                     match_aspect = True)

        self.img_data_source = ColumnDataSource(data=dict(
            x  = [],
            y  = [],
            dw = [],
            dh = [],
            f  = [],
        ))
        color_mapper = LogColorMapper(palette="Viridis256")
        self.image_glyph = fig.image(source = self.img_data_source, image = 'f', x = 'x', y = 'y', dw = 'dw', dh = 'dh', color_mapper=color_mapper)


        ## self.rect_peak_data_source_0 = ColumnDataSource(data=dict(
        ##     x      = [],
        ##     y      = [],
        ##     width  = [],
        ##     height = [],
        ## ))
        self.rect_peak_data_source_1 = ColumnDataSource(data=dict(
            x      = [],
            y      = [],
            width  = [],
            height = [],
        ))
        ## fig.rect(source     =  self.rect_peak_data_source_0,
        ##          x          =  'x',
        ##          y          =  'y',
        ##          width      =  'width',
        ##          height     =  'height',
        ##          line_width = 1.0,
        ##          line_color = 'yellow',
        ##          fill_color = None)
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
                event = new[0]

                img_and_peaks_dict = cxi_peakdiff.get_img_and_peaks_by_event(event)
                img     = img_and_peaks_dict["image"]
                ## peaks_0 = img_and_peaks_dict["peaks_0"]
                peaks_1 = img_and_peaks_dict["peaks_1"]

                ## self.update_image_panel(img, peaks_0, peaks_1)
                self.update_image_panel(img, peaks_1)

        scatter_plot_data_source.selected.on_change('indices', load_selected)

        self.selected_event_div = fig


    def init_indices_panel(self):
        fig_height   = self.fig_height
        fig_width    = self.fig_width
        title_width  = 30
        title_height = fig_height

        scatter_plot_data_source = self.scatter_plot_data_source

        # [[[ CONTENT -- SELECTED INDICES ]]]
        # CustomJS callback to update Div on selection
        selected_indices_div = Div(width  = 2 * fig_width,
                                   height = title_height,
                                   text   = "")
        callback = CustomJS(args=dict(source=scatter_plot_data_source, div=selected_indices_div, title_height=title_height), code="""
            const inds = source.selected.indices;
            let text = "<div style='height:" + title_height + "px; overflow-y: scroll;'>";  // Ensure this div has a set height and overflow-y property
            for (let i = 0; i < inds.length; i++) {
                text += inds[i] + "\\n";
            }
            text += "</div>";
            div.text = text;
        """)

        scatter_plot_data_source.selected.js_on_change('indices', callback)

        self.selected_indices_div = selected_indices_div


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
        section_div['selected_indices'] = Div(text = div_text, width = title_width, height = title_height)
        section_div['selected_indices'].margin = (0, 5, 0, 0)

        self.section_div = section_div


    def init_layout(self):
        fig                  = self.fig
        selected_event_div   = self.selected_event_div
        selected_indices_div = self.selected_indices_div
        section_div          = self.section_div

        layout = {}
        layout['scatter_plot'    ] = row(section_div['scatter_plot']    , gridplot([[fig['n_peaks'], fig['match_rates']]]))
        layout['selected_event'  ] = row(section_div['selected_event']  , selected_event_div)
        layout['selected_indices'] = row(section_div['selected_indices'], selected_indices_div)

        final_layout = column(layout['scatter_plot'], layout['selected_event'], layout['selected_indices'])

        self.final_layout = final_layout


    def run(self):
        curdoc().add_root(self.final_layout)
