# Indexing

A powerful data extraction, transformation, and vector search library for AI applications.

## Overview

Indexing is a Python library that provides a complete pipeline for extracting data from various sources, transforming it into embeddings, loading it into vector databases, and performing semantic searches. It's designed for AI applications that require knowledge retrieval capabilities.

## Core Components

### 1. Extractors

Extractors fetch data from various sources and prepare it for processing:

```python
from indexing.extractor import GitHubRepoExtractor

# Extract code from a GitHub repository
extractor = GitHubRepoExtractor(
    repo_url="https://github.com/username/repo",
    branch_name="main",
    pat="your_personal_access_token"  # Optional for public repos
)

# Extract the data
records = await extractor.extract()
```

Supported extractors:
- `GitHubRepoExtractor` - Extract code from GitHub repositories
- `GitLabRepoExtractor` - Extract code from GitLab repositories
- `AzureDevopsRepoExtractor` - Extract code from Azure DevOps repositories
- `QuipExtractor` - Extract documents from Quip

#### Local File Indexing

For indexing local files without using an extractor, you can directly use the `FileContent` class:

```python
from indexing.serializers import FileContent, SourceItemKind
import hashlib
import uuid

# Create FileContent objects for local files
file_records = []

# Read a local file
with open('/path/to/local/file.py', 'r') as f:
    content = f.read()
    
# Create a FileContent object
file_content = FileContent(
    path='file.py',
    content=content,
    version_tag='1.0',
    provider_item_id='local',
    # Optional parameters
    checksum=hashlib.sha256(content.encode()).hexdigest(),
    uuid_str=str(uuid.uuid4()),
    kind=SourceItemKind.file.value
)

# Convert to dictionary format expected by transformers
file_records.append(file_content.to_dict())
```

### 2. Transformers

Transformers convert extracted data into embeddings and prepare it for indexing:

```python
from indexing.transformers import RepoTransformer
from indexing.embeddings import EmbeddingGenerator

# Initialize embedding generator
embedder = EmbeddingGenerator()

# Transform repository data
transformer = RepoTransformer()
documents = await transformer.transform(
    records=records,
    embedder=embedder,
    knowledge_base_id="kb-123"
)
```

Supported transformers:
- `RepoTransformer` - Transforms code repositories
- `QuipTransformer` - Transforms Quip documents

### 3. Loaders

Loaders insert the transformed data into vector databases:

```python
from indexing.loaders import ElasticSearchLoader
from indexing.db import ElasticSearchVectorDb

# Initialize database connection
db = ElasticSearchVectorDb()
db.elastic_search_url = ["http://localhost:9200"]

# Load data into Elasticsearch
loader = ElasticSearchLoader(db)
await loader.load({
    "index_name": "my_vector_index",
    "documents": documents,
    "batch_size": 50
})
```

### 4. Vector Search

Perform semantic searches on the indexed data:

```python
from indexing.serializers import VectorSearchRequest

# Create a search request
request = VectorSearchRequest(
    query="How does the authentication system work?",
    knowledge_base_id=["kb-123"],
    top_answer_count=5
)

# Perform the search
results = await db.search_content(
    request=request,
    index_name="my_vector_index",
    embedding_generator=embedder
)

# Process results
for result in results:
    print(f"Score: {result['_score']}")
    print(f"Content: {result['content']}")
    print("---")
```

## Installation

```bash
pip install indexing
```

## Requirements

- Python 3.9+
- Dependencies:
  - httpx
  - beautifulsoup4
  - pydantic
  - asyncio
  - uuid
  - tiktoken
  - elasticsearch

## Complete Pipeline Example

### Remote Repository Pipeline

```python
import asyncio
from indexing.extractor import GitHubRepoExtractor
from indexing.transformers import RepoTransformer
from indexing.loaders import ElasticSearchLoader
from indexing.db import ElasticSearchVectorDb
from indexing.embeddings import EmbeddingGenerator
from indexing.serializers import VectorSearchRequest

async def index_and_search():
    # 1. Extract data
    extractor = GitHubRepoExtractor(
        repo_url="https://github.com/username/repo",
        branch_name="main"
    )
    records = await extractor.extract()
    
    # 2. Transform data
    embedder = EmbeddingGenerator()
    transformer = RepoTransformer()
    documents = await transformer.transform(
        records=records,
        embedder=embedder,
        knowledge_base_id="kb-123"
    )
    
    # 3. Load data
    db = ElasticSearchVectorDb()
    db.elastic_search_url = ["http://localhost:9200"]
    loader = ElasticSearchLoader(db)
    await loader.load({
        "index_name": "my_vector_index",
        "documents": documents
    })
    
    # 4. Search data
    request = VectorSearchRequest(
        query="How does the authentication system work?",
        knowledge_base_id=["kb-123"],
        top_answer_count=5
    )
    results = await db.search_content(
        request=request,
        index_name="my_vector_index",
        embedding_generator=embedder
    )
    
    return results

# Run the pipeline
results = asyncio.run(index_and_search())
```

### Local Files Pipeline

```python
import asyncio
import os
from indexing.serializers import FileContent
from indexing.transformers import RepoTransformer
from indexing.loaders import ElasticSearchLoader
from indexing.db import ElasticSearchVectorDb
from indexing.embeddings import EmbeddingGenerator
from indexing.serializers import VectorSearchRequest

async def index_local_files():
    # 1. Prepare local file records
    file_records = []
    local_dir = '/path/to/local/files'
    
    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.endswith('.py'):  # Filter files as needed
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                relative_path = os.path.relpath(file_path, local_dir)
                file_content = FileContent(
                    path=relative_path,
                    content=content,
                    version_tag='local-1.0',
                    provider_item_id='local'
                )
                file_records.append(file_content.to_dict())
    
    # 2. Transform data
    embedder = EmbeddingGenerator()
    transformer = RepoTransformer()
    documents = await transformer.transform(
        records=file_records,
        embedder=embedder,
        knowledge_base_id="kb-local"
    )
    
    # 3. Load data
    db = ElasticSearchVectorDb()
    db.elastic_search_url = ["http://localhost:9200"]
    loader = ElasticSearchLoader(db)
    await loader.load({
        "index_name": "local_files_index",
        "documents": documents
    })
    
    # 4. Search data
    request = VectorSearchRequest(
        query="What does this code do?",
        knowledge_base_id=["kb-local"],
        top_answer_count=5
    )
    results = await db.search_content(
        request=request,
        index_name="local_files_index",
        embedding_generator=embedder
    )
    
    return results

# Run the pipeline
results = asyncio.run(index_local_files())
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Piyush Tyagi - piyushtyagi28@hotmail.com

Project Link: https://github.com/piyush1856/indexing
