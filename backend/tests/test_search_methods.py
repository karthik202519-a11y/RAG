"""Additional search modes, graph provenance, and exact retrieved context for generation."""
import os
import pytest
from sqlalchemy import select
from studio.config import settings
from studio.models import GraphBuild,GraphEntity,GraphEdge,IndexChunk
from studio import db as dbmod
from studio.vectorstores import STORES
from test_indexes import build,make_docs
from test_generation import configured,FakeChat
from studio.generation import service as gen
from studio.retrieval import pipeline


@pytest.mark.parametrize("store",list(STORES))
def test_tfidf_graph_and_fusion_on_every_store(client,auth,project,monkeypatch,store):
    from studio.vectorstores import chroma_store,lancedb_store,pgvector_store
    monkeypatch.setattr(chroma_store,"_client",None);monkeypatch.setattr(lancedb_store,"_db",None);monkeypatch.setattr(lancedb_store,"_meta",{})
    monkeypatch.setattr(pgvector_store,"_engine",None)
    monkeypatch.setattr(settings,"pgvector_url",os.getenv("TEST_PGVECTOR_URL","postgresql://postgres:studio@localhost:5434/vec"))
    if not STORES[store].status()[0]:pytest.skip("store unavailable")
    docs=make_docs(client,auth,project);idx=build(client,auth,project,store=store)
    base=f"/indexes/{idx['id']}"
    missing=client.post(base+'/search',headers=auth,json={"query":"Apollo","mode":"graph"})
    assert missing.status_code==409
    with dbmod.SessionLocal() as db:
        chunk=db.scalars(select(IndexChunk).where(IndexChunk.index_id==idx['id'],IndexChunk.document_id==docs['apollo.txt']).order_by(IndexChunk.chunk_index)).first()
        a=GraphEntity(index_id=idx['id'],name='Saturn V',norm='saturn v',etype='rocket')
        b=GraphEntity(index_id=idx['id'],name='Apollo',norm='apollo',etype='program')
        db.add_all([a,b,GraphBuild(index_id=idx['id'],status='ready',model='simulated-extraction')]);db.flush()
        db.add(GraphEdge(index_id=idx['id'],src_id=a.id,dst_id=b.id,relation='carried',chunk_id=chunk.id));db.commit()
    for mode in ('tfidf','graph','graph_hybrid'):
        r=client.post(base+'/search',headers=auth,json={"query":"Apollo","mode":mode})
        assert r.status_code==200,r.text
        assert r.json()['results'][0]['filename']=='apollo.txt'
        assert r.json()['results'][0]['citation']
        empty=client.post(base+'/search',headers=auth,json={"query":"Apollo","mode":mode,"document_ids":["missing"]})
        assert empty.status_code==200 and empty.json()['results']==[]
    proof=client.get(base+'/construction',headers=auth).json()
    assert proof['graph']['edges'][0]['source']=='Saturn V'
    assert 'Saturn V' in proof['graph']['edges'][0]['text']
    assert proof['graph']['edges'][0]['chunk_id']


def test_answer_uses_exact_pipeline_results_without_searching_again(client,auth,project,configured,monkeypatch):
    make_docs(client,auth,project);idx=build(client,auth,project)
    ret=client.post(f"/indexes/{idx['id']}/search",headers=auth,json={"query":"Chlorophyll","mode":"bm25"}).json()
    monkeypatch.setattr(pipeline,'run',lambda *a,**kw:ret)
    from studio import retrieval
    monkeypatch.setattr(retrieval,'search',lambda *a,**kw:pytest.fail('Answer must use pipeline evidence, not run another search'))
    chat=FakeChat('Chlorophyll absorbs red and blue light [1].');monkeypatch.setattr(gen,'chat_complete',chat)
    r=client.post(f"/indexes/{idx['id']}/answer",headers=auth,json={"question":"What absorbs red and blue light?","rerank":"cross_encoder"})
    assert r.status_code==200,r.text
    assert 'Chlorophyll' in chat.calls[0]['messages'][1]['content']
    assert r.json()['context_passages'][0]['chunk_id']==ret['results'][0]['chunk_id']
