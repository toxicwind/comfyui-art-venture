from .nodes import GetSAMEmbedding, SAMEmbeddingToImage, LamaLoader, LamaInpaint

NODE_CLASS_MAPPINGS = {
    "GetSAMEmbedding": GetSAMEmbedding,
    "SAMEmbeddingToImage": SAMEmbeddingToImage,
    "LamaLoader": LamaLoader,
    "LamaInpaint": LamaInpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetSAMEmbedding": "Get SAM Embedding",
    "SAMEmbeddingToImage": "SAM Embedding to Image",
    "LamaLoader": "Lama Loader",
    "LamaInpaint": "Lama Remove Object",
}
