# Activity 2: Goal-Driven Autonomy

## Verification and Reflection

### 1. Did the agent stay focused on the goal throughout the loop?

Yes. The agent remained focused on the goal during each iteration of the loop. It continuously referenced the original objective and the previously generated steps. This helped the agent produce a sequence of related actions that contributed to achieving the goal of deploying a secure web application for a small business.

### 2. Challenge: Modify the loop to perform 5 steps instead of 3. How does this affect the detail of the plan?

To generate 5 steps, change:

```python
max_steps = 3
```

to:

```python
max_steps = 5
```

Increasing the number of steps provides a more detailed plan. Instead of combining multiple tasks into a few broad actions, the agent can break the objective into smaller and more specific steps. This results in clearer guidance and a more organized approach to completing the project.
