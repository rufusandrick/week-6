from __future__ import annotations

from typing import Any

import numpy as np
import triton_python_backend_utils as pb_utils

NAME_INPUT_PROB = "probabilities"
NAME_OUTPUT_PROB = "probability"


class TritonPythonModel:
    def initialize(self, _args: dict[str, Any] | None = None) -> None:
        """Initialize Triton model instance."""

    def execute(self, requests: list[pb_utils.InferenceRequest]) -> list[pb_utils.InferenceResponse]:
        """Produce responses with class-one probabilities."""
        responses: list[pb_utils.InferenceResponse] = []
        for request in requests:
            proba_tensor = pb_utils.get_input_tensor_by_name(request, NAME_INPUT_PROB)
            probas = proba_tensor.as_numpy()

            class1 = probas[:, 1:2].astype(np.float32, copy=False)
            output_tensor = pb_utils.Tensor(NAME_OUTPUT_PROB, class1)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self) -> None:
        """Cleanup hook invoked when the model is unloaded."""
