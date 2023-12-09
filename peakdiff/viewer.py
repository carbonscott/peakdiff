#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bokeh.models   import ColumnDataSource, Circle, Div, CustomJS, Span, HoverTool
from bokeh.plotting import figure, show
from bokeh.layouts  import gridplot, row, column
from bokeh.io       import curdoc


class Viewer:

    def __init__(self, data_source, fig_height = 600, fig_width = 600):
        self.fig_height  = fig_height
        self.fig_width   = fig_width
        self.data_source = ColumnDataSource(data_source)

        self.init_scatterplot_panel()
        self.init_indices_panel()
        self.init_section_title_panel()
        self.init_layout()


    def init_scatterplot_panel(self):
        fig_height   = self.fig_height
        fig_width    = self.fig_width
        title_width  = 30
        title_height = fig_height

        data_source = self.data_source

        TOOLS = "box_select,lasso_select,wheel_zoom,pan,reset,help,"

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
                                                 source                  = data_source,
                                                 size                    = 10,
                                                 fill_color              = "blue",
                                                 line_color              =  None,
                                                 fill_alpha              = 0.5,
                                                 nonselection_fill_alpha = 0.005,
                                                 nonselection_fill_color = "blue")
        scatter_match_rates = fig['match_rates'].scatter('match_rates_x',
                                                         'match_rates_y',
                                                         source                  = data_source,
                                                         size                    = 10,
                                                         fill_color              = "red",
                                                         line_color              =  None,
                                                         fill_alpha              = 0.5,
                                                         nonselection_fill_alpha = 0.005,
                                                         nonselection_fill_color = "red")

        self.fig = fig


    def init_indices_panel(self):
        fig_height   = self.fig_height
        fig_width    = self.fig_width
        title_width  = 30
        title_height = fig_height

        data_source = self.data_source

        # [[[ CONTENT -- SELECTED INDICES ]]]
        # CustomJS callback to update Div on selection
        selected_indices_div = Div(width  = 2 * fig_width,
                                   height = title_height,
                                   text   = "")
        callback = CustomJS(args=dict(source=data_source, div=selected_indices_div, title_height=title_height), code="""
            const inds = source.selected.indices;
            let text = "<div style='height:" + title_height + "px; overflow-y: scroll;'>";  // Ensure this div has a set height and overflow-y property
            for (let i = 0; i < inds.length; i++) {
                text += inds[i] + "\\n";
            }
            text += "</div>";
            div.text = text;
        """)

        data_source.selected.js_on_change('indices', callback)

        self.selected_indices_div = selected_indices_div


    def init_section_title_panel(self):
        fig_height   = self.fig_height
        fig_width    = self.fig_width
        title_width  = 30
        title_height = fig_height

        data_source = self.data_source

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
        selected_indices_div = self.selected_indices_div
        section_div          = self.section_div

        layout = {}
        layout['scatter_plot'    ] = row(section_div['scatter_plot'], gridplot([[fig['n_peaks'], fig['match_rates']]]))
        layout['selected_indices'] = row(section_div['selected_indices'], selected_indices_div)

        final_layout = column(layout['scatter_plot'], layout['selected_indices'])

        self.final_layout = final_layout


    def run(self):
        curdoc().add_root(self.final_layout)
