from pymilvus import MilvusClient, DataType
import json


# ✅ Step 1: Load Embedded Chunks
def load_embedded_chunks(file_path):
    """
    Load embedded text chunks from a JSONL file.
    """
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} embedded chunks from {file_path}.")
    return chunks


# ✅ Step 2: Connect to Milvus
def connect_to_milvus():
    """
    Establish a connection to the Milvus server.
    """
    try:
        client = MilvusClient(
            uri="http://127.0.0.1:19530",
            token="root:Milvus"
        )
        print("Successfully connected to Milvus.")
        return client
    except Exception as e:
        print(f"Connection failed: {e}")
        return None


# ✅ Step 3: Create Collection Schema
def create_milvus_collection(client):
    """
    Define and create a collection schema in Milvus.
    """
    try:
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="Interview Book Chunks with BGE Embeddings"
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=2048
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024
        )
        schema.add_field(
            field_name="metadata",
            datatype=DataType.JSON
        )
        client.create_collection(
            collection_name="interview_book_bge",
            schema=schema
        )
        print("Collection 'interview_book_bge' created successfully.")
    except Exception as e:
        print(f"Collection creation failed: {e}")


# ✅ Step 4: Insert Data into Milvus
def insert_chunks_into_milvus(client, chunks):
    """
    Insert embedded chunks into Milvus.
    """
    try:
        data = []
        for chunk in chunks:
            data.append({
                "id": int(chunk["id"]),
                "text": chunk["text"],
                "embedding": chunk["embedding"],
                "metadata": chunk["metadata"]
            })

        client.insert(
            collection_name="interview_book_bge",
            data=data
        )

        client.flush(collection_name="interview_book_bge")
        print(f"Inserted {len(chunks)} chunks into Milvus and flushed data successfully.")
    except Exception as e:
        print(f"Data insertion failed: {e}")


# ✅ Step 5: Create Index on Embeddings
def create_index(client):
    """
    Create an index on the embedding field for optimized search.
    """
    try:
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",  # Balanced choice for precision and performance
            metric_type="COSINE",
            params={
                "nlist": 2048  # Number of clusters (optimal range: 128–2048)
            }
        )
        # 6. Create indexes
        client.create_index(
            collection_name="interview_book_bge",
            index_params=index_params,
            sync=False
        )
        print("Index created successfully on 'embedding' field.")
    except Exception as e:
        print(f"Failed to create index: {e}")


# ✅ Step 6: Validate Index
def validate_index(client):
    """
    Validate the status of the index.
    """
    try:
        indexes = client.list_indexes(collection_name="interview_book_bge")
        if indexes:
            print("Indexes in collection:")
            for index in indexes:
                print(index)
        else:
            print("No indexes found on the collection.")
    except Exception as e:
        print(f"Failed to validate indexes: {e}")


# ✅ Step 7: Load Collection
def load_collection(client):
    """
    Load the collection into memory after index creation.
    """
    try:
        client.load_collection(
            collection_name="interview_book_bge"
        )
        print("Collection loaded successfully and ready for querying.")
    except Exception as e:
        print(f"Failed to load collection: {e}")


# ✅ Step 8: Verify Data in Milvus
def verify_milvus_data(client):
    """
    Verify that data has been successfully inserted into Milvus.
    """
    try:
        stats = client.get_collection_stats(collection_name="interview_book_bge")
        count = stats.get("row_count", 0)
        print(f"Total Chunks in Milvus: {count}")
    except Exception as e:
        print(f"Failed to verify data: {e}")


# ✅ Step 9: Describe Collection Schema
def describe_milvus_collection(client):
    """
    Describe the schema of the Milvus collection.
    """
    try:
        schema = client.describe_collection("interview_book_bge")
        print("Collection Schema:")
        print(schema)
    except Exception as e:
        print(f"Failed to describe collection: {e}")

# ✅ Step 10: Main Function
def main():
    input_file = 'data/embedded_interview_book_bge.jsonl'
    chunks = load_embedded_chunks(input_file)
    client = connect_to_milvus()
    
    if client is None:
        return
    
    client.drop_collection(
    collection_name="interview_book_bge"
    )

    # Check if collection already exists
    if "interview_book_bge" not in client.list_collections():
        create_milvus_collection(client)
    
    insert_chunks_into_milvus(client, chunks)
    create_index(client)
    validate_index(client)
    load_collection(client)
    verify_milvus_data(client)
    describe_milvus_collection(client)

# ✅ Execute the Script
if __name__ == '__main__':
    main()
