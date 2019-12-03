import plotly.graph_objs as go
import pandas as pd

df = pd.read_csv("hand-parcoords.csv");

fig = go.Figure(
    data =
        go.Parcoords(
            line=dict(color=df['colour'],
                      colorscale = 'Rainbow',
                      showscale = True,
                      cmin = 1,
                      cmax = 5.5),
            dimensions=list([
                dict(label='Thu-PP/DP', values=df['Thu-PP/DP']),
                dict(label='Thu-MC/DP', values=df['Thu-MC/DP']),

                dict(label='Ind-IP/DP', values=df['Ind-IP/DP']),
                dict(label='Ind-PP/DP', values=df['Ind-PP/DP']),
                dict(label='Ind-MC/DP', values=df['Ind-MC/DP']),

                dict(label='Mid-IP/DP', values=df['Mid-IP/DP']),
                dict(label='Mid-PP/DP', values=df['Mid-PP/DP']),
                dict(label='Mid-MC/DP', values=df['Mid-MC/DP']),

                dict(label='Ring-IP/DP', values=df['Ring-IP/DP']),
                dict(label='Ring-PP/DP', values=df['Ring-PP/DP']),
                dict(label='Ring-MC/DP', values=df['Ring-MC/DP']),

                dict(label='Lit-IP/DP', values=df['Lit-IP/DP']),
                dict(label='Lit-PP/DP', values=df['Lit-PP/DP']),
                dict(label='Lit-MC/DP', values=df['Lit-MC/DP']),
                            ]))
                )
fig.show()
