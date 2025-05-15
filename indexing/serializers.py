import enum
import hashlib
import uuid

class SourceItemKind(enum.Enum):
    file = "file"
    document = "document"
    folder = "folder"
    message = "message"

class KnowledgeBaseDocument:
    def __init__(self, id, title, embedding, content, chunk_references, source, is_chunked, is_public, description, knowledge_base_id, metadata):
        self.id = id
        self.title = title
        self.embedding = embedding
        self.content = content
        self.chunk_references = chunk_references
        self.source = source
        self.is_chunked = is_chunked
        self.is_public = is_public
        self.description = description
        self.knowledge_base_id = knowledge_base_id
        self.metadata = metadata

    def to_dict(self):
        """Convert the object to a dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "embedding": self.embedding,
            "content": self.content,
            "chunk_references": self.chunk_references,
            "source": self.source,
            "is_chunked": self.is_chunked,
            "is_public": self.is_public,
            "description": self.description,
            "knowledge_base_id": self.knowledge_base_id,
            "metadata": self.metadata if self.metadata else None
        }


class FileContent:
    def __init__(
        self,
        path: str,
        content: str,
        version_tag: str,
        provider_item_id: str,
        checksum: str = None,
        uuid_str: str = None,
        kind: str = SourceItemKind.file.value
    ):
        self.path = path
        self.content = content
        self.version_tag = version_tag
        self.provider_item_id = provider_item_id
        self.checksum = checksum or hashlib.sha256(content.encode()).hexdigest()
        self.uuid = uuid_str or str(uuid.uuid4())
        self.kind = kind

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "content": self.content,
            "version_tag": self.version_tag,
            "provider_item_id": self.provider_item_id,
            "checksum": self.checksum,
            "uuid": self.uuid,
            "kind": self.kind
        }

class VectorSearchRequest:
    query: str
    knowledge_base_id: list[int]
    matching_percentage: float
    top_answer_count: int