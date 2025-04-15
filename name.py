try:
    from chromadb import PersistentClient
    print("✅ PersistentClient is available!")
except ImportError:
    print("❌ PersistentClient is NOT available in this version.")

