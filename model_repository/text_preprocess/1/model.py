import os

import cloudpickle
import numpy as np
import triton_python_backend_utils as pb_utils

NAME_INPUT_TEXT_FIELD = "text"


class TritonPythonModel:
    def initialize(self, args) -> None:
        transform_path = os.path.join(os.path.dirname(__file__), "transform.pkl")
        with open(transform_path, "rb") as f:
            self.transform = cloudpickle.load(f)

    def execute(self, requests):
        """Здесь для простоты мы просто переиспользуем часть пайплайна, который отвечает за предобработку текста.
        Так как на выходе всегда получаются векторы одинаковой длины, то пользуемся здесь простым итеративным подходом.
        :param requests:
        :return:
        """
        responses = []
        for request in requests:
            input_text = pb_utils.get_input_tensor_by_name(request, NAME_INPUT_TEXT_FIELD).as_numpy()
            decoded_texts = [text.decode("utf-8") for text in input_text.flatten()]
            encoded_input = self.transform.transform(decoded_texts).toarray().astype(np.float32)
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("X", encoded_input)
            ])
            responses.append(inference_response)
        return responses

    def finalize(self) -> None:
        pass
