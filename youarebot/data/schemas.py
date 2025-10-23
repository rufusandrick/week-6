from pydantic import UUID4, BaseModel, StrictStr


class IncomingMessage(BaseModel):
    """Входная схема одного сообщения, которое нужно сохранить
    и на основании которого проводится классификация диалога.
    """

    text: StrictStr
    dialog_id: UUID4
    id: UUID4
    participant_index: int


class Prediction(BaseModel):
    """Результат классификации:
    - id: уникальный идентификатор предсказания
    - message_id: UUID сообщения, на которое мы отвечаем
    - dialog_id: ID диалога
    - participant_index: индекс участника
    - is_bot_probability: вероятность, что в диалоге присутствует бот.
    """

    id: UUID4
    message_id: UUID4
    dialog_id: UUID4
    participant_index: int
    is_bot_probability: float
