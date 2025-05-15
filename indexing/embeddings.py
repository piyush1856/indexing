from openai import OpenAI

class EmbeddingGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-ada-002"

    def generate_embedding(self, text: str):
        try:
            response = self.client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding  # Access as an attribute, not as a dictionary
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")