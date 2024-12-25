from pymilvus import MilvusClient, DataType
from datetime import datetime

# ✅ Milvus Client Initialization
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# ✅ Define Collection Schema
def create_collection():
    """
    Create a Milvus collection for interview questions if it doesn't exist.
    """
    client.drop_collection(
    collection_name="interview_questions"
    )

    if "interview_questions" not in client.list_collections():
        schema = client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="Interview Questions Collection"
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=1024
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024  # Vector dimension
        )
        client.create_collection(
            collection_name="interview_questions",
            schema=schema
        )
        print("✅ Milvus collection 'interview_questions' created successfully.")
    else:
        print("🔄 Collection 'interview_questions' already exists.")


# ✅ Insert Questions into Milvus
def insert_questions():
    """
    Insert interview questions into Milvus with placeholder embeddings.
    """
    interview_questions = [
        "Tell me about yourself.",
        "Why do you want to work here?",
        "What are your greatest strengths?",
        "What are your weaknesses?",
        "Where do you see yourself in five years?",
        "Describe a time when you had to overcome a significant challenge at work.",
        "Why are you leaving your current job?",
        "Tell me about a time when you worked as part of a team.",
        "Describe a situation where you had to work independently.",
        "How do you handle stress and pressure?",
        "What is your greatest professional achievement?",
        "Tell me about a time when you failed.",
        "How do you prioritize your work?",
        "Why should we hire you?",
        "Describe a time when you had to manage a conflict at work.",
        "What motivates you?",
        "How do you handle criticism?",
        "What are your salary expectations?",
        "What do you know about our company?",
        "Do you have any questions for us?"
    ]

    # Placeholder embedding: 1024-dimensional zero vector
    placeholder_embedding = [0.0] * 1024

    # Prepare data for insertion
    data = []
    for i, question in enumerate(interview_questions):
        data.append({
            "id": i + 1,
            "text": question,
            "embedding": placeholder_embedding
        })

    try:
        client.insert(
            collection_name="interview_questions",
            data=data
        )
        client.flush(collection_name="interview_questions")

        print(f"✅ Successfully inserted {len(data)} questions into Milvus with placeholder embeddings.")

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",  # Balanced choice for precision and performance
            metric_type="L2",
        )
        client.create_index(
            collection_name="interview_questions",
            index_params=index_params,
            sync=False
        )

        indexes = client.list_indexes(collection_name="interview_questions")
        if indexes:
            print("Indexes in collection:")
            for index in indexes:
                print(index)
        else:
            print("No indexes found on the collection.")
        
    except Exception as e:
        print(f"❌ Failed to insert questions: {e}")


# ✅ Verify Data Insertion
def verify_data():
    """
    Verify the data inserted into Milvus.
    """
    try:
        stats = client.get_collection_stats(collection_name="interview_questions")
        print(f"📊 Total Records in Collection: {stats.get('row_count', 0)}")
    except Exception as e:
        print(f"❌ Failed to verify collection stats: {e}")


# ✅ Main Workflow
def main():
    create_collection()
    insert_questions()
    verify_data()


# ✅ Execute Script
if __name__ == "__main__":
    main()
