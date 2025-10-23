from __future__ import annotations

from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import triton_python_backend_utils as pb_utils

NAME_INPUT_TEXT_FIELD = "text"
NAME_OUTPUT_FEATURES = "X"


class TritonPythonModel:
    def initialize(self, _args: dict[str, Any] | None = None) -> None:
        """Load the serialized text preprocessing pipeline."""
        transform_path = Path(__file__).parent / "transform.pkl"
        with transform_path.open("rb") as file:
            self.transform = cloudpickle.load(file)

    def execute(self, requests: list[pb_utils.InferenceRequest]) -> list[pb_utils.InferenceResponse]:
        """Transform incoming text requests into numerical features.

        The preprocessing pipeline produces arrays with a consistent shape,
        allowing sequential processing of batched requests.
        """
        responses: list[pb_utils.InferenceResponse] = []
        for request in requests:
            input_text = pb_utils.get_input_tensor_by_name(request, NAME_INPUT_TEXT_FIELD).as_numpy()
            decoded_texts = [text.decode("utf-8") for text in input_text.ravel()]
            encoded_input = self.transform.transform(decoded_texts).toarray().astype(np.float32)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor(NAME_OUTPUT_FEATURES, encoded_input)]
            )
            responses.append(inference_response)
        return responses

    def finalize(self) -> None:
        """Cleanup hook invoked when the model is unloaded."""
