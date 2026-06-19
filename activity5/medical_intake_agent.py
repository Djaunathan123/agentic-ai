import os
import json
from enum import Enum
from typing import List

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"  


# ---------------------------------------------------------------------------
# 1. Schema Specifications
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class Symptom(BaseModel):
    symptom_name: str = Field(..., description="Name of the symptom, e.g. 'stomach cramping'")
    severity: Severity = Field(..., description="Severity level: LOW, MEDIUM, or HIGH")
    duration_days: int = Field(..., ge=0, description="How many days the symptom has persisted")

class MedicalIntake(BaseModel):
    symptoms: List[Symptom] = Field(..., description="List of reported symptoms")
    allergies: List[str] = Field(default_factory=list, description="List of known allergies, empty list if none")
    urgency_rating: int = Field(..., ge=1, le=10, description="Triage urgency rating from 1 (low) to 10 (critical)")
    clinical_reasoning: str = Field(
        ..., description="Chain-of-thought explanation justifying the urgency rating and severity selections"
    )

class IntakeGenerationError(Exception):
    """Raised when the agent fails to produce a valid MedicalIntake after all retries."""
    pass

# ---------------------------------------------------------------------------
# 2. Self-Correction Loop
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are a clinical intake assistant. Read the patient's free-text description "
    "and extract a structured medical intake record. Always populate every field. "
    "When the patient states an explicit urgency/pain number (e.g. '15 out of 10'), "
    "use that exact number as the urgency_rating, even if it falls outside a normal "
    "1-10 scale — report faithfully what the patient said rather than adjusting it "
    "yourself. Provide detailed, step-by-step clinical_reasoning that explains how "
    "you arrived at the severity levels and the overall urgency_rating."
)

def process_intake(patient_input: str, max_retries: int = 3) -> MedicalIntake:
    """
    Calls the Gemini API with a constrained JSON schema, validates the response
    against the MedicalIntake Pydantic model, and self-corrects on failure by
    feeding the validation error back to the model.
    """
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=f"Patient description:\n\n{patient_input}")],
        )
    ]

    last_error = None

    for attempt in range(1, max_retries + 1):
        print(f"\n[Attempt {attempt}/{max_retries}] Calling Gemini API...")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=MedicalIntake,
            ),
        )

        raw_text = response.text

        try:
            data = json.loads(raw_text)
            record = MedicalIntake.model_validate(data)
            print(f"[Attempt {attempt}] Validation succeeded.")
            return record

        except (ValidationError, json.JSONDecodeError) as e:
            last_error = e
            print(f"[Attempt {attempt}] Validation FAILED:\n{e}")

            # Append the model's bad output and our error feedback to the
            # conversation history so the next call can self-correct.
            contents.append(
                types.Content(role="model", parts=[types.Part(text=raw_text)])
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                "Your previous response failed schema validation with the "
                                f"following error:\n\n{e}\n\n"
                                "Please correct the JSON output so it strictly satisfies the "
                                "MedicalIntake schema (in particular: urgency_rating must be an "
                                "integer between 1 and 10 inclusive, duration_days must be >= 0, "
                                "and severity must be one of LOW, MEDIUM, HIGH). "
                                "Return the corrected, complete JSON object only."
                            )
                        )
                    ],
                )
            )

    raise IntakeGenerationError(
        f"Failed to produce a valid MedicalIntake after {max_retries} attempts. "
        f"Last error: {last_error}"
    )

# ---------------------------------------------------------------------------
# 3. Adversarial Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_input = (
        "My stomach is cramping incredibly badly since last night! The pain is "
        "unbearable, definitely an urgency of 15 out of 10! I don't think I have allergies."
    )

    try:
        record = process_intake(test_input)
        print("\n--- Validated Intake Record ---")
        print(record.model_dump_json(indent=2))
    except Exception as e:
        print(f"Failed: {e}")