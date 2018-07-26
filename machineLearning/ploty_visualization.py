import plotly

plotly.tools.set_credentials_file(username='pauldipin', api_key='uyVMAumqAkawB7bkxodT')
import plotly.graph_objs as go


# %matplotlib inline
class PlotlyVisualization:
    @staticmethod
    def createHistPlotForVarForBinaryClass1(data, class_variable, numeric_variable, label_zero_meaning, label_1_meaning,
                                            title):
        # Make a separate list
        x1 = list(data[data[class_variable] == 0][numeric_variable])
        x2 = list(data[data[class_variable] == 1][numeric_variable])

        trace1 = go.Histogram(
            x=x1,
            name=label_zero_meaning,
            # xbins=dict(
            #     start=0,
            #     end=100
            # ),
            marker=dict(
                color='#FFD7E9',
            ),
            opacity=0.75
        )
        trace2 = go.Histogram(
            x=x2,
            name=label_1_meaning,
            # xbins=dict(
            #     start=0,
            #     end=100
            # ),
            marker=dict(
                color='#EB89B5'
            ),
            opacity=0.75
        )
        data = [trace1, trace2]

        layout = go.Layout(
            title=title,
            xaxis=dict(
                title=numeric_variable
            ),
            yaxis=dict(
                title='Frequency'
            ),
            bargap=0.2,
            bargroupgap=0.1
        )
        fig = go.Figure(data=data, layout=layout)
        return fig

    @staticmethod
    def createHistPlotForVarForBinaryClass2(data, class_variable, numeric_variable, label_zero_meaning, label_1_meaning,
          title):
        # Make a separate list
        x1 = list(data[data[class_variable] == 0][numeric_variable])
        x2 = list(data[data[class_variable] == 1][numeric_variable])
        trace1 = go.Histogram(
            x=x1,
            opacity=0.75,
            name=label_zero_meaning
        )
        trace2 = go.Histogram(
            x=x2,
            opacity=0.75,
            name=label_1_meaning,
        )
        data = [trace1, trace2]
        layout = go.Layout(barmode='overlay', title=title,
                           xaxis=dict(
                               title=numeric_variable
                           ))
        fig = go.Figure(data=data, layout=layout)
        return fig
