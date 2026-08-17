from pydantic import BaseModel


class DraftedReminder(BaseModel):
    subject: str
    body: str
