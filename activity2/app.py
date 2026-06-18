import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

planner_identity = """
You are a Strategic Project Planner.
Your goal is to take a high-level objective and break it into exactly 3 sequential steps.
Respond ONLY in a numbered list.
"""

def autonomous_planner(goal):
    print(f"\n[GOAL]: {goal}")
    print("-" * 30)

    steps_taken = 0
    max_steps = 3
    plan = []

    while steps_taken < max_steps:
        print(f"[REASONING] Planning Step {steps_taken + 1}...")

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config={"system_instruction": planner_identity},
            contents=f"Goal: {goal}. Current steps completed: {plan}. Analyze the goal and provide Step {steps_taken + 1}:"
        )

        step_description = response.text.strip()
        plan.append(step_description)
        steps_taken += 1

    return plan

my_goal = "Deploy a secure web application for a small business."
final_plan = autonomous_planner(my_goal)

print("\n--- FINAL AUTONOMOUS PLAN ---")
for i, step in enumerate(final_plan, 1):
    print(f"STEP {i}: {step}")