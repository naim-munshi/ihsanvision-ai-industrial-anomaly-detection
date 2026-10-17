"""Public API response contracts. Scores are raw anomaly values, not probabilities."""
from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat, NonNegativeFloat


class PredictionResponse(BaseModel):
    """Validated single-image inference result for the FastAPI/OpenAPI boundary."""

    model: Literal["padim", "patchcore"] = Field(
        description="Anomaly detection model family used for inference."
    )
    category: str = Field(
        default="bottle",
        description="Operator-registered industrial product category used for inference.",
    )
    anomaly_score: FiniteFloat = Field(
        description="Raw uncalibrated anomaly score; NOT a probability."
    )
    predicted_anomaly: bool | None = Field(
        default=None,
        description="Boolean at the saved checkpoint's operating threshold, if available.",
    )
    latency_ms: NonNegativeFloat | None = Field(
        default=None,
        description="Model prediction span in milliseconds; excludes HTTP and UI overhead.",
    )
    heatmap_png_base64: str | None = Field(
        default=None, description="Optional base64 PNG of relative, not calibrated localization."
    )
    model_reused: bool | None = Field(
        default=None,
        description="True when an already-restored checkpoint was reused in this server process.",
    )
    notes: str | None = Field(default=None, description="Warnings about score semantics.")
    inspection_id: str | None = Field(
        default=None,
        description="Private operator history record ID only when opt-in persistence is enabled.",
    )


class InspectionReviewUpdate(BaseModel):
    """A single operator review decision, independent of the model output."""

    verdict: Literal["confirmed_normal", "confirmed_anomaly", "needs_followup"] | None = Field(
        description="Final human assessment, follow-up request, or null to clear review."
    )

    model_config = {"extra": "forbid"}
