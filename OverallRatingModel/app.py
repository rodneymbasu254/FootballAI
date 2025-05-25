import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

model = joblib.load('football_lr_model.pkl')

def predict_overall_rating(age, potential, weak_foot, skill, crossing, finishing, heading_accuracy, short_passing, volleys, dribbling, curve, freekick_accuracy, long_passing, ball_control, acceleration, sprint_speed, agility, reactions, balance, shot_power, jumping, stamina, strength, long_shots, aggression, interceptions, positioning, vision, penalties, composure, marking, standing_tackle, sliding_tackle
):
    input_data = np.array([[age, potential, weak_foot, skill, crossing, finishing, heading_accuracy, short_passing, volleys, dribbling, curve, freekick_accuracy, long_passing, ball_control, acceleration, sprint_speed, agility, reactions, balance, shot_power, jumping, stamina, strength, long_shots, aggression, interceptions, positioning, vision, penalties, composure, marking, standing_tackle, sliding_tackle
]])
    prediction = model.predict(input_data)[0][0]
    return round(prediction, 2)

def radar_chart(values):
    labels = ["Speed", "Dribbling", "Shooting", "Passing", "Strength", "Stamina"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax.fill(angles, values, color='blue', alpha=0.3)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    return fig

def combined_interface(age, potential, weak_foot, skill, crossing, finishing, heading_accuracy, short_passing, volleys, dribbling, curve, freekick_accuracy, long_passing, ball_control, acceleration, sprint_speed, agility, reactions, balance, shot_power, jumping, stamina, strength, long_shots, aggression, interceptions, positioning, vision, penalties, composure, marking, standing_tackle, sliding_tackle
):
    rating = predict_overall_rating(age, potential, weak_foot, skill, crossing, finishing, heading_accuracy, short_passing, volleys, dribbling, curve, freekick_accuracy, long_passing, ball_control, acceleration, sprint_speed, agility, reactions, balance, shot_power, jumping, stamina, strength, long_shots, aggression, interceptions, positioning, vision, penalties, composure, marking, standing_tackle, sliding_tackle
)
    radar = radar_chart([sprint_speed, dribbling, shot_power, short_passing, strength, stamina])
    return rating, radar

demo = gr.Interface(
    fn=combined_interface,
    inputs=[
        gr.Slider(16, 45, step=1, label="Age"),
        gr.Slider(40, 99, step=1, label="Potential"),
        gr.Slider(1, 5, step=1, label="Weak Foot"),
        gr.Slider(1, 5, step=1, label="Skill Moves"),
        gr.Slider(30, 99, step=1, label="Crossing"),
        gr.Slider(30, 99, step=1, label="Finishing"),
        gr.Slider(30, 99, step=1, label="Heading"),
        gr.Slider(30, 99, step=1, label="Short_Passing"),
        gr.Slider(30, 99, step=1, label="Volleys"),
        gr.Slider(30, 99, step=1, label="Dribbling"),
        gr.Slider(30, 99, step=1, label="Curve"),
        gr.Slider(30, 99, step=1, label="Freekick"),
        gr.Slider(30, 99, step=1, label="Long Passing"),
        gr.Slider(30, 99, step=1, label="Ball Control"),
        gr.Slider(30, 99, step=1, label="Acceleration"),
        gr.Slider(30, 99, step=1, label="Sprint Speed"),
        gr.Slider(30, 99, step=1, label="Agility"),
        gr.Slider(30, 99, step=1, label="Reactions"),
        gr.Slider(30, 99, step=1, label="Balance"),
        gr.Slider(30, 99, step=1, label="Shot Power"),
        gr.Slider(30, 99, step=1, label="Jumping"),
        gr.Slider(30, 99, step=1, label="Stamina"),
        gr.Slider(30, 99, step=1, label="Strength"),
        gr.Slider(30, 99, step=1, label="Long Shots"),
        gr.Slider(30, 99, step=1, label="Aggression"),
        gr.Slider(30, 99, step=1, label="Interception"),
        gr.Slider(30, 99, step=1, label="Positioning"),
        gr.Slider(30, 99, step=1, label="Vision"),
        gr.Slider(30, 99, step=1, label="Penalties"),
        gr.Slider(30, 99, step=1, label="Composure"),
        gr.Slider(30, 99, step=1, label="Marking"),
        gr.Slider(30, 99, step=1, label="Standing Tackle"),
        gr.Slider(30, 99, step=1, label="Sliding Tackle")
    ],
    outputs=[
        gr.Textbox(label="Predicted Overall Rating"),
        gr.Plot(label="Player Attributes Radar Chart")
    ],
    title="Football AI Player Rating Predictor",
    description="Adjust the sliders to set a player's attributes and get an AI-predicted overall rating!"
)

demo.launch()
