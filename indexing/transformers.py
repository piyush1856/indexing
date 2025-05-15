from abc import ABC, abstractmethod
import tiktoken
import asyncio
from bs4 import BeautifulSoup

import re
from indexing.serializers import KnowledgeBaseDocument
from indexing.embeddings import EmbeddingGenerator


class DataTransformer(ABC):
    """Abstract base class for data transformation."""

    @abstractmethod
    async def transform(self, records: list[dict], embedder: EmbeddingGenerator, **kwargs) -> list[dict]:
        """Transforms extracted records into a structured format."""
        pass

class QuipTransformer(DataTransformer):
    """Transforms Quip HTML documents into structured format with embeddings."""

    async def transform(self, records: list[dict], embedder: EmbeddingGenerator, **kwargs) -> list[dict]:
        tasks = [self.convert_to_document(record, embedder, **kwargs) for record in records]
        results = await asyncio.gather(*tasks)
        # Flatten the list of lists and convert to dicts
        return [doc.to_dict() for sublist in results for doc in sublist]

    async def convert_to_document(self, record: dict, embedder: EmbeddingGenerator, **kwargs) -> list[
        KnowledgeBaseDocument]:

        # Parse HTML content with improved handling
        html_content = record["content"]
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove invisible elements and clean up whitespace
        for element in soup.find_all(string=lambda text: isinstance(text, str) and (
                '\u200b' in text or '\u00a0' in text)):  # Remove ZWSP and NBSP
            element.replace_with(' ' if '\u00a0' in element else '')

        # Extract text content with better structure preservation
        # Keep list markers and formatting
        for li in soup.find_all('li'):
            # Add bullet points for list items
            if li.get('value') == '1' and li.parent.name == 'ul':
                li.insert(0, '• ')
            else:
                li.insert(0, '• ')

        # Convert bold text to markdown-style bold
        for bold in soup.find_all('b'):
            bold.replace_with(f"**{bold.get_text()}**")

        # Get clean text with proper spacing
        text_content = soup.get_text(separator='\n').strip()

        # Remove consecutive blank lines
        text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
        
        # Prepare for chunking
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text_content)

        chunk_size = 8000
        start = 0
        chunks = []
        line_offsets = []
        
        # Split content into lines for tracking
        lines = text_content.splitlines()
        
        # Precompute token offsets per line
        token_offsets_per_line = [len(encoding.encode(line + "\n")) for line in lines]
        token_line_indices = []
        token_index = 0
        for i, token_count in enumerate(token_offsets_per_line):
            token_line_indices.extend([i + 1] * token_count)
            token_index += token_count

        # Create chunks based on token size
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = encoding.decode(tokens[start:end])
            chunks.append(chunk_text)

            start_line = token_line_indices[start] if start < len(token_line_indices) else None
            end_line = token_line_indices[end - 1] if end - 1 < len(token_line_indices) else None
            line_offsets.append((start_line, end_line))
            start = end

        chunk_ids = [f"{record['uuid']}#{i + 1}" for i in range(len(chunks))]
        documents = []

        for i, chunk_text in enumerate(chunks):
            comment = f"""
            The following content belongs to the Quip document at path: {record['path']}.
            This is chunk {i + 1} out of {len(chunks)} total chunks in the document.
            The chunk spans lines {line_offsets[i][0]} to {line_offsets[i][1]}.
            """
            chunk_text = f"{comment}\n{chunk_text}"

            start_line, end_line = line_offsets[i]

            doc = KnowledgeBaseDocument(
                id=chunk_ids[i],
                title=record['path'],
                embedding=embedder.generate_embedding(chunk_text),
                content=chunk_text,
                chunk_references=chunk_ids,
                source="quip",
                is_chunked=len(chunks) > 1,
                is_public=False,
                description="",
                knowledge_base_id=kwargs.get("knowledge_base_id"),
                metadata={
                    "chunk_number": i + 1,
                    "total_chunks": len(chunks),
                    "path": record['path'],
                    "start_line": start_line,
                    "end_line": end_line,
                    "source_url": record['source_url']
                }
            )
            documents.append(doc)

        return documents

class RepoTransformer(DataTransformer):
    """Transforms Azure DevOps repository data into structured format with embeddings."""

    async def transform(self, records: list[dict], embedder: EmbeddingGenerator, **kwargs) -> list[dict]:
        tasks = [self.convert_to_document(record, embedder, **kwargs) for record in records]
        results = await asyncio.gather(*tasks)
        # Flatten the list of lists and convert to dicts
        return [doc.to_dict() for sublist in results for doc in sublist]

    async def convert_to_document(self, record: dict, embedder: EmbeddingGenerator, **kwargs) -> list[
        KnowledgeBaseDocument]:

        code = record["content"]
        lines = code.splitlines()
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(code)

        chunk_size = 8000
        start = 0
        chunks = []
        line_offsets = []
        current_token_index = 0

        # Precompute token offsets per line
        token_offsets_per_line = [len(encoding.encode(line + "\n")) for line in lines]
        token_line_indices = []
        token_index = 0
        for i, token_count in enumerate(token_offsets_per_line):
            token_line_indices.extend([i + 1] * token_count)
            token_index += token_count

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = encoding.decode(tokens[start:end])
            chunks.append(chunk_text)

            start_line = token_line_indices[start] if start < len(token_line_indices) else None
            end_line = token_line_indices[end - 1] if end - 1 < len(token_line_indices) else None
            line_offsets.append((start_line, end_line))
            start = end

        chunk_ids = [f"{record['uuid']}#{i + 1}" for i in range(len(chunks))]
        documents = []

        for i, chunk_text in enumerate(chunks):
            comment = f"""
            The following content belongs to the file at path: {record['path']}.
            This is chunk {i + 1} out of {len(chunks)} total chunks in the document.
            The chunk spans lines {line_offsets[i][0]} to {line_offsets[i][1]}.
            """
            chunk_text = f"{comment}\n{chunk_text}"

            start_line, end_line = line_offsets[i]

            doc = KnowledgeBaseDocument(
                id=chunk_ids[i],
                title=record['path'],
                embedding=embedder.generate_embedding(chunk_text),
                content=chunk_text,
                chunk_references=chunk_ids,
                source="azure_devops",
                is_chunked=len(chunks) > 1,
                is_public=False,
                description="",
                knowledge_base_id=kwargs.get("knowledge_base_id"),
                metadata={
                    "chunk_number": i + 1,
                    "total_chunks": len(chunks),
                    "path": record['path'],
                    "start_line": start_line,
                    "end_line": end_line
                }
            )
            documents.append(doc)

        return documents

