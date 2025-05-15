from abc import ABC, abstractmethod
from typing import Any, Dict

class DataLoader(ABC):
    def __init__(self, db):
        self.db = db

    @abstractmethod
    async def load(self, params: Dict[str, Any]):
        """Load data into the vector database. The params dict should contain index_name and other DB-specific fields."""
        pass

class ElasticSearchLoader(DataLoader):
    KNOWLEDGE_BASE_INDEXING_SETTINGS = {
        "settings": {
            "number_of_shards": 5,
            "number_of_replicas": 2
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "standard"},
                "content": {
                    "type": "text",
                    "index": False,
                    "store": False
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": 1536,
                    "index": True,
                    "similarity": "cosine"
                },
                "chunk_references": {"type": "keyword"},
                "source": {"type": "keyword"},
                "is_chunked": {"type": "boolean"},
                "is_public": {"type": "boolean"},
                "description": {"type": "text"},
                "knowledge_base_id": {"type": "keyword"},
                "metadata": {
                    "type": "object",
                    "dynamic": True
                }
            }
        }
    }

    async def load(self, params: Dict[str, Any]):
        """Loads data into Elasticsearch by creating an index and performing a bulk insert in batches."""
        try:
            await self.db.connect()

            index_name = params.get("index_name")
            documents = params.get("documents", [])
            settings = params.get("settings", self.KNOWLEDGE_BASE_INDEXING_SETTINGS)
            batch_size = params.get("batch_size", 50)

            if not index_name:
                raise ValueError("Missing 'index_name' in parameters.")

            # Function to split documents into smaller batches
            def chunks(lst, size):
                for i in range(0, len(lst), size):
                    yield lst[i:i + size]

            total_success = 0
            all_failed = []

            # Loop through documents in batches
            for batch in chunks(documents, batch_size):
                # Construct bulk actions for the current batch
                actions = [
                    {
                        "_index": index_name,
                        "_source": doc,
                        "_id": doc.get("id")
                    }
                    for doc in batch
                ]

                # Perform bulk insert for the current batch
                result = await self.db.bulk_insert(index_name, batch, settings, actions)

                # Accumulate successes and failures
                total_success += result.get("success", 0)
                all_failed.extend(result.get("failed", []))

            final_result = {
                "success": total_success,
                "failed": all_failed
            }

            return final_result

        except Exception as e:
            raise e

        finally:
            await self.db.close()