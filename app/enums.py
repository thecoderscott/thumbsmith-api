from enum import StrEnum, IntEnum

class StyleEnum(StrEnum):
    game = "game"
    photo = "photo"
    avatar = "avatar"

class StrengthEnum(IntEnum):
    weak = 1
    normal = 2
    strong = 3