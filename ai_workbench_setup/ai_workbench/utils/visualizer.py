import plotly.graph_objects as go
import pandas as pd
from utils.logger import setup_logger

class Visualizer:
    def __init__(self):
        self.logger = setup_logger(__name__)

    def generate_visualizations(self, df: pd.DataFrame) -> dict:
        try:
            visualizations = {
                "bar_charts": {},
                "radar_plot": {},
                "table": df.to_dict(orient="records")
            }

            for metric in df.columns:
                if metric == "model":
                    continue
                fig = go.Figure(data=[
                    go.Bar(
                        x=df["model"],
                        y=df[metric],
                        text=df[metric].round(3),
                        textposition="auto"
                    )
                ])
                fig.update_layout(
                    title=f"{metric.upper()} Comparison",
                    xaxis_title="Model",
                    yaxis_title=metric,
                    template="plotly_dark"
                )
                visualizations["bar_charts"][metric] = fig.to_json()

            metrics = [col for col in df.columns if col != "model"]
            fig = go.Figure()
            for _, row in df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[metric] for metric in metrics],
                    theta=metrics,
                    fill='toself',
                    name=row["model"]
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                title="Model Comparison (Radar)",
                template="plotly_dark"
            )
            visualizations["radar_plot"] = fig.to_json()

            self.logger.info("Generated visualizations")
            return visualizations
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            raise