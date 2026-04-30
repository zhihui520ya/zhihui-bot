# memory/chroma_store.py
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from config import DATA_DIR

class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path, device="cpu")

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

class VectorStore:
    """向量存储，所有 ChromaDB 操作通过专属线程池执行，避免阻塞事件循环。"""

    def __init__(self, persist_dir=None, model_path="./bge_model"):
        if persist_dir is None:
            persist_dir = os.path.join(DATA_DIR, "vector_store")
        self.client = chromadb.PersistentClient(path=persist_dir)
        abs_model_path = os.path.abspath(model_path)
        print(f"模型绝对路径: {abs_model_path}")
        self.embed_fn = LocalEmbeddingFunction(abs_model_path)
        # 单线程执行器，确保 ChromaDB 操作串行化（线程安全）
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def _run(self, func, *args, **kwargs):
        """在专属线程池中同步执行 ChromaDB 操作。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs)
        )

    def _get_collection_sync(self, session_id):
        """同步版 get_collection，给线程池内部调用。"""
        return self.client.get_or_create_collection(
            name=session_id, embedding_function=self.embed_fn
        )

    async def add_message(self, session_id, msg_id, text, metadata=None):
        await self._run(
            lambda: self._get_collection_sync(session_id).add(
                ids=[msg_id], documents=[text], metadatas=[metadata or {}]
            )
        )

    async def retrieve(self, session_id, query, k=5, where=None):
        def _do_retrieve():
            col = self._get_collection_sync(session_id)
            if col.count() == 0:
                return [], []
            results = col.query(query_texts=[query], n_results=k, where=where)
            return results['documents'][0], results['metadatas'][0]
        return await self._run(_do_retrieve)

    async def add_summary(self, session_id, summary_id: int, text: str, metadata: dict = None):
        await self._run(
            lambda: self._get_collection_sync(session_id).add(
                ids=[str(summary_id)], documents=[text], metadatas=[metadata or {}]
            )
        )

    async def get_all_ids(self, session_id) -> List[str]:
        def _do_get_all():
            col = self._get_collection_sync(session_id)
            if col.count() == 0:
                return []
            results = col.get()
            return results['ids']
        return await self._run(_do_get_all)

    async def get_by_ids(self, session_id, ids: List[str]) -> List[Dict[str, Any]]:
        def _do_get_by_ids():
            col = self._get_collection_sync(session_id)
            results = col.get(ids=ids)
            items = []
            for i, doc in enumerate(results['documents']):
                items.append({
                    'id': results['ids'][i],
                    'document': doc,
                    'metadata': results['metadatas'][i] if results['metadatas'] else {}
                })
            return items
        return await self._run(_do_get_by_ids)

    async def delete_by_ids(self, session_id, ids: List[str]):
        def _do_delete():
            col = self._get_collection_sync(session_id)
            col.delete(ids=ids)
        await self._run(_do_delete)

    async def delete_collection(self, session_id):
        def _do_delete():
            try:
                self.client.delete_collection(session_id)
            except ValueError:
                pass
        await self._run(_do_delete)

    async def get_max_id(self, session_id) -> int:
        ids = await self.get_all_ids(session_id)
        if not ids:
            return 0
        int_ids = []
        for id in ids:
            try:
                int_ids.append(int(id))
            except:
                continue
        return max(int_ids) if int_ids else 0


_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
