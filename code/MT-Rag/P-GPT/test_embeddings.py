from langchain_huggingface import HuggingFaceEmbeddings
import time

def test_huggingface_embeddings():
    print("Testing HuggingFace Embeddings with AkshaySandbox/pregnancy-mpnet-embeddings...")
    
    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="AkshaySandbox/pregnancy-mpnet-embeddings")
    
    # Test text
    test_text = "What are the common symptoms of pregnancy in the first trimester?"
    
    # Time the embedding generation
    start_time = time.time()
    embedding = embeddings.embed_query(test_text)
    end_time = time.time()
    
    # Print results
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    return embedding

if __name__ == "__main__":
    test_huggingface_embeddings()