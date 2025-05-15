from abc import ABC, abstractmethod
from elasticsearch import AsyncElasticsearch
import asyncio
from elasticsearch.helpers import async_bulk
from typing import Dict, Any, Optional

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