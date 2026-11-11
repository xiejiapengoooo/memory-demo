from pathlib import Path

from pymilvus import DataType, MilvusClient


def create_client(dimension: int) -> MilvusClient:
    path = Path(".memory/memory.db")
    path.parent.mkdir(parents=True, exist_ok=True)

    client = MilvusClient(uri=str(path))

    try:
        if not client.has_collection("memory"):
            schema = MilvusClient.create_schema(
                auto_id=True, enable_dynamic_field=False
            )

            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)

            schema.add_field(
                field_name="text", datatype=DataType.VARCHAR, max_length=8000
            )

            schema.add_field(
                field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension
            )

            schema.add_field(
                field_name="entities",
                datatype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                default_value=[],
            )

            client.create_collection(
                collection_name="memory", schema=schema, metric_type="COSINE"
            )

        return client

    except Exception:
        client.close()
        raise
