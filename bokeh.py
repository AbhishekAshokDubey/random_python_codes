
from bokeh.io import output_file, show, vplot
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import HoverTool
from collections import OrderedDict
from bokeh.models import LinearAxis, Range1d
from bokeh.models import NumeralTickFormatter

data = pd.read_csv(r"C:\Users\JSreekantan\Desktop\bokeh\well_logging.csv", parse_dates=['DATE_TIME'])

x = data.DATE_TIME
y1 = data.DEPT
y2 = data.HKLD
y3 = data.BPOS
y4 = data.RPM
y5 = data.SPPA
y6 = data.STOR

#output_notebook()

p1 = figure(plot_width=1200, plot_height=300,
           tools="pan,box_zoom,reset,resize,save,crosshair,hover", 
           title="Drilling Analytics",
           y_axis_label='',
		   x_axis_type = "datetime",
           toolbar_location="left"
          )

p2 = figure(plot_width=1200, plot_height=300,
           tools="pan,box_zoom,reset,resize,save,crosshair,hover", 
           y_axis_label='',
		   x_axis_type = "datetime",
           toolbar_location="left"
          )

p3 = figure(plot_width=1200, plot_height=300,
           tools="pan,box_zoom,reset,resize,save,crosshair,hover", 
           y_axis_label='',
		   x_axis_type = "datetime",
           toolbar_location="left"
          )

p4 = figure(plot_width=1200, plot_height=300,
           tools="pan,box_zoom,reset,resize,save,crosshair,hover", 
           y_axis_label='',
		   x_axis_type = "datetime",
           toolbar_location="left"
          )  
          
p5 = figure(plot_width=1200, plot_height=300,
           tools="pan,box_zoom,reset,resize,save,crosshair,hover", 
           y_axis_label='',
           x_axis_label = 'Date',
		   x_axis_type = "datetime",
           toolbar_location="left"
          )          

hover1 = p1.select(dict(type=HoverTool))
hover1.tooltips = OrderedDict([('x-tip', '@x'),('y-tip', '@y')])

hover2 = p2.select(dict(type=HoverTool))
hover2.tooltips = OrderedDict([('x-tip', '@x'),('y-tip', '@y')])

hover3 = p3.select(dict(type=HoverTool))
hover3.tooltips = OrderedDict([('x-tip', '@x'),('y-tip', '@y')])

hover4 = p4.select(dict(type=HoverTool))
hover4.tooltips = OrderedDict([('x-tip', '@x'),('y-tip', '@y')])

hover5 = p5.select(dict(type=HoverTool))
hover5.tooltips = OrderedDict([('x-tip', '@x'),('y-tip', '@y')])

output_notebook()

output_file("timeseries.html")

p1.line(x, y1, legend="Weight")

#p2.yaxis.formatter = NumeralTickFormatter(format="0.0%")

p2.line(x, y2, legend="Muscle Mass", line_color="red")
p3.line(x, y3, legend="Muscle Mass", line_color="red")
p4.line(x, y4, legend="Muscle Mass", line_color="red")

p5.extra_y_ranges = {"foo": Range1d(start= min(y6), end=max(y6))}
p5.add_layout(LinearAxis(y_range_name="foo"), 'right')
p5.line(x, y5, legend="SPPA")
p5.line(x, y6, legend="STOR", line_color="red", y_range_name="foo")

p = vplot(p1,p2,p3,p4,p5)

show(p)
