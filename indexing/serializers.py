import enum
class SourceItemKind(enum.Enum):
    file = "file"
    document = "document"
    folder = "folder"
    message = "message"