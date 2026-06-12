from visual_memory_system.retrieval.embedding import EmbeddingRetriever
from visual_memory_system.schema import MemoryUnit, QueryRecord


def test_embedding_retriever_returns_requested_topk() -> None:
    units = [
        MemoryUnit(unit_id="u1", text="a"),
        MemoryUnit(unit_id="u2", text="b"),
        MemoryUnit(unit_id="u3", text="c"),
    ]
    retriever = EmbeddingRetriever(
        memory_embeddings={
            "u1": [1.0, 0.0],
            "u2": [0.9, 0.1],
            "u3": [0.0, 1.0],
        },
        query_embeddings={"q1": [1.0, 0.0]},
    )
    query = QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q")
    assert retriever.retrieve(query, units, topk=2) == ["u1", "u2"]

