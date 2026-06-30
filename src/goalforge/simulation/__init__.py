"""Monte-Carlo match simulation.

Draws each team's goal total from the scoreline model, then allocates goals and
assists to individual players, aggregating over many runs into score / scorer /
assister probabilities. Supports vectorized CPU and (optionally) GPU execution.
"""
