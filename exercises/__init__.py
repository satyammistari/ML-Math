"""
exercises/__init__.py
"""
from exercises.ex01_linear_algebra import exercises as ex01
from exercises.ex03_calculus import exercises as ex03
from exercises.ex05_probability import exercises as ex05
from exercises.ex08_backprop import exercises as ex08
from exercises.ex10_supervised import exercises as ex10
from exercises.ex12_unsupervised import exercises as ex12
from exercises.ex15_rl import exercises as ex15
from exercises.ex17_deep_learning import exercises as ex17

all_exercises = {
    "01 Linear Algebra":     ex01,
    "03 Calculus":           ex03,
    "05 Probability":        ex05,
    "08 Backprop":           ex08,
    "10 Supervised":         ex10,
    "12 Unsupervised":       ex12,
    "15 Reinforcement":      ex15,
    "17 Deep Learning":      ex17,
}
