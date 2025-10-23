from pydantic import UUID4, BaseModel, StrictStr


class IncomingMessage(BaseModel):
    """Incoming message schema used for dialog classification."""

    text: StrictStr
    dialog_id: UUID4
    id: UUID4
    participant_index: int


class Prediction(BaseModel):
    """Prediction result with dialog metadata and bot probability."""

    id: UUID4
    message_id: UUID4
    dialog_id: UUID4
    participant_index: int
    is_bot_probability: float
