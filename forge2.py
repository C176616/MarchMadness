import plotly.express as px
import plotly.io as pio

pio.renderers.deafult = "notebook"
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.show()