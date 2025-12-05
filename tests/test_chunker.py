from qi_align.chunk.chunker import generate_chunks_from_chain, Chunk

def test_chunk_generation():
    chain = [(0,0), (100000,100000)]
    chunks = list(generate_chunks_from_chain(chain, chunk_size=20000, overlap=2000))

    assert len(chunks) > 1

    for ch in chunks:
        assert isinstance(ch, Chunk)
        assert ch.ref_end > ch.ref_start
        assert ch.qry_end > ch.qry_start
