from app.embeddings import HashEmbedder
import numpy as np

def cos(a,b): return float(np.dot(a,b))
def test_hash_embedder_cosine():
    emb = HashEmbedder(dim=128)
    a = emb.embed_one("alice near lighthouse")
    b = emb.embed_one("lighthouse by the sea")
    c = emb.embed_one("stone bridge over river")
    # ближе к lighthouse, чем к bridge
    assert cos(a,b) > cos(a,c)
