import plotly
import plotly.graph_objects as go
import os

def plot_lr_test(model_name:str, model_results:dict):

	lr = model_results["lr"]
	loss = model_results["loss"]

	fig = go.Figure()

	fig.add_trace(go.Scatter(
		x=lr, 
		y=loss,
		mode='lines',
	))

	fig.update_layout(
        title=f"{model_name} Learning Rate vs Loss Graph", 
        xaxis_title='Learning Rate',
        yaxis_title='Loss',
        yaxis=dict(range=(min(loss)-0.5,min(loss)+2)),
        xaxis=dict(type='log'),
    )

	os.makedirs('./plots', exist_ok=True)
	fig.write_image(f"./plots/{model_name}.png")

	fig.show()