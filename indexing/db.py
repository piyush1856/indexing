from abc import ABC, abstractmethod
from elasticsearch import AsyncElasticsearch
import asyncio
from elasticsearch.helpers import async_bulk
from typing import Dict, Any, Optional, List

from indexing.serializers import VectorSearchRequest


class VectorDb(ABC):
    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    async def bulk_insert(self, index_name: str, documents: list[dict], mapping: dict, actions: list):
        pass

class ElasticSearchVectorDb(VectorDb):
    def __init__(self):
        self.client = None
        self.elastic_search_url: list[str] = []

    async def connect(self, retries=3, delay=2) -> None:
        """Connect to Elasticsearch with retries."""
        for attempt in range(retries):
            try:
                self.client = AsyncElasticsearch(hosts=self.elastic_search_url)
                await self.client.info()
                return
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise ConnectionError(f"Failed to connect to Elasticsearch: {str(e)}")

    async def close(self) -> None:
        if self.client:
            await self.client.close()

    async def create_index(self, index_name: str, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            if await self.client.indices.exists(index=index_name):
                return {"acknowledged": False, "message": "Index already exists"}

            return await self.client.indices.create(index=index_name, body=settings)
        except Exception as e:
            raise Exception(f"Failed to create index: {str(e)}")

    async def bulk_insert(self, index_name: str, documents: list[dict], mapping: dict, actions: list):
        """Ensures index exists and inserts multiple documents into Elasticsearch using async bulk API."""
        await self.create_index(index_name=index_name, settings=mapping)

        # Perform bulk indexing asynchronously
        success, failed = await async_bulk(self.client, actions, raise_on_error=False)

        # Extract failed _id values
        failed_ids = [failure['index']['_id'] for failure in failed if 'index' in failure]

        return {"success": success, "failed": failed_ids}

    async def search_content(self, request: VectorSearchRequest, index_name: str, embedding_generator) -> List[dict]:
        """
        Perform a KNN similarity search and return concatenated chunks (max 3) for each result,
        including chunk_num-1 and chunk_num+1 if available. Filters by knowledge_base_id.
        """
        try:
            await self.connect()
            # Generate query embedding
            query_vector = await asyncio.to_thread(embedding_generator.generate_embedding, request.query)

            if len(query_vector) != 1536:
                raise ValueError(f"Query vector has incorrect dimensions: {len(query_vector)} (Expected: 1536)")

            num_candidates = max(request.top_answer_count * 5, 100)

            # KNN Search query
            knn_query = {
                "size": request.top_answer_count,
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": request.top_answer_count,
                    "num_candidates": num_candidates,
                    "filter": [
                        {"terms": {"knowledge_base_id": request.knowledge_base_id}}
                    ]
                },
                "_source": ["id", "content", "is_chunked"]
            }

            response = await self.client.search(index=index_name, body=knn_query)
            hits = response["hits"]["hits"]

            results = []

            for hit in hits:
                doc = hit["_source"]
                doc_id = doc.get("id")
                is_chunked = doc.get("is_chunked", False)

                if not is_chunked:
                    results.append({
                        "content": doc.get("content"),
                        "_score": hit["_score"]
                    })
                    continue

                # Process chunked document
                if "#" not in doc_id:
                    continue  # skip invalid format

                logical_id, chunk_str = doc_id.rsplit("#", 1)

                try:
                    chunk_num = int(chunk_str)
                except ValueError:
                    continue  # skip if chunk number is not integer

                # Attempt to fetch chunk-1 and chunk+1
                adjacent_ids = [
                    f"{logical_id}#{chunk_num - 1}",
                    f"{logical_id}#{chunk_num + 1}"
                ]

                # Build query to fetch additional chunks
                chunk_query = {
                    "size": 2,
                    "query": {
                        "bool": {
                            "filter": [
                                {"terms": {"id": adjacent_ids}}
                            ]
                        }
                    },
                    "_source": ["id", "content"]
                }

                adjacent_response = await self.client.search(index=index_name, body=chunk_query)
                adjacent_chunks = {d["_source"]["id"]: d["_source"]["content"] for d in
                                   adjacent_response["hits"]["hits"]}

                # Order the chunks correctly: [chunk-1, current, chunk+1]
                full_content = ""
                if f"{logical_id}#{chunk_num - 1}" in adjacent_chunks:
                    full_content += adjacent_chunks[f"{logical_id}#{chunk_num - 1}"] + "\n"
                full_content += doc.get("content", "")
                if f"{logical_id}#{chunk_num + 1}" in adjacent_chunks:
                    full_content += "\n" + adjacent_chunks[f"{logical_id}#{chunk_num + 1}"]

                results.append({
                    "content": full_content.strip(),
                    "_score": hit["_score"]
                })

            return results

        except Exception as e:
            raise Exception(f"Failed to perform KNN search and fetch content: {str(e)}")
        finally:
            await self.close()