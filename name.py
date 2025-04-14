from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")
print(model.encode("test sample")[:5])  