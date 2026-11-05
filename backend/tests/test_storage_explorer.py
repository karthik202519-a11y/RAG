"""Read-only, scoped table samples and actual native records in every vector store."""
import os
import pytest
from sqlalchemy import select
from studio import db as dbmod
from studio.models import User
from studio.db import Base
from studio.config import settings
from studio.vectorstores import STORES
from conftest import upload,wait_for_job,register
from test_indexes import build,make_docs


@pytest.mark.parametrize("store",list(STORES))
def test_native_records_and_chunk_links_for_every_store(client,auth,project,monkeypatch,store):
    from studio.vectorstores import chroma_store,lancedb_store,pgvector_store
    monkeypatch.setattr(chroma_store,"_client",None);monkeypatch.setattr(lancedb_store,"_db",None);monkeypatch.setattr(lancedb_store,"_meta",{})
    monkeypatch.setattr(pgvector_store,"_engine",None)
    monkeypatch.setattr(settings,"pgvector_url",os.getenv("TEST_PGVECTOR_URL","postgresql://postgres:studio@localhost:5434/vec"))
    if not STORES[store].status()[0]:pytest.skip("store unavailable")
    docs=make_docs(client,auth,project);idx=build(client,auth,project,store=store)
    r=client.get(f"/indexes/{idx['id']}/storage?sample=2",headers=auth)
    assert r.status_code==200,r.text
    out=r.json()
    assert out['store']==store and out['stored_vectors']==idx['chunk_count']
    assert out['native_schema'] and out['objects'] and len(out['samples'])==2
    for row in out['samples']:
        assert row['vector']['dimension']==256 and len(row['vector']['first_values'])==12
        assert row['vector']['norm']==pytest.approx(1,abs=1e-5)
        assert row['id']==row['linked_chunk']['id']
        assert row['linked_chunk']['document_id'] in docs.values()
        assert row['metadata']['document_id']==row['linked_chunk']['document_id']
    if store=='pgvector':assert any('hnsw' in x['definition'] for x in out['native_indexes'])
    if store=='faiss':assert all(x['exists'] for x in out['objects'])
    other=register(client,email=f"foreign-{store}@example.com")
    assert client.get(f"/indexes/{idx['id']}/storage",headers=other).status_code==404


def test_all_tables_purposes_links_samples_and_secret_redaction(client,auth,project):
    make_docs(client,auth,project);build(client,auth,project)
    with dbmod.SessionLocal() as db:
        user=db.scalar(select(User).where(User.email=='a@example.com'))
        user.password_hash='NEVER_EXPOSE_PASSWORD_HASH_123';db.commit()
    r=client.get(f"/projects/{project['id']}/storage?sample=2",headers=auth)
    assert r.status_code==200,r.text
    out=r.json();by={t['name']:t for t in out['tables']}
    assert set(Base.metadata.tables).issubset(by) and 'alembic_version' in by
    assert all(t['purpose'] and t['benefit'] and t['columns'] for t in out['tables'])
    assert 'NEVER_EXPOSE_PASSWORD_HASH_123' not in r.text
    assert by['users']['samples'][0]['password_hash']=='<redacted>'
    assert by['projects']['samples'][0]['id']==project['id']
    assert by['index_chunks']['samples'] and by['embedding_cache']['samples']
    assert any(x['source_table']=='index_chunks' and x['target_table']=='indexes' and x['kind']=='foreign_key' for x in out['relationships'])
    assert any(x['target_table']=='vector record' and x['kind']=='logical' for x in out['relationships'])
    assert all(len(t['samples'])<=2 for t in out['tables'])


def test_other_project_rows_and_foreign_accounts_are_not_sampled(client,auth,project):
    owned=make_docs(client,auth,project)
    ws=client.get('/workspaces',headers=auth).json()[0]['id']
    other=client.post('/projects',headers=auth,json={"workspace_id":ws,"name":"OTHER_PROJECT_SENTINEL_123"}).json()
    doc=upload(client,auth,other['id'],'NEVER_OTHER_PROJECT_123.txt',b'private other-project text').json()['results'][0]
    assert wait_for_job(client,auth,doc['job_id'])['status']=='succeeded'
    r=client.get(f"/projects/{project['id']}/storage",headers=auth)
    assert r.status_code==200
    assert 'OTHER_PROJECT_SENTINEL_123' not in r.text and 'NEVER_OTHER_PROJECT_123' not in r.text
    focused=client.get(f"/projects/{project['id']}/storage?focus_table=documents&focus_id={owned['apollo.txt']}",headers=auth).json()
    rows=next(t['samples'] for t in focused['tables'] if t['name']=='documents')
    assert len(rows)==1 and rows[0]['id']==owned['apollo.txt']
    forbidden=client.get(f"/projects/{project['id']}/storage?focus_table=documents&focus_id={doc['document_id']}",headers=auth).json()
    assert next(t['samples'] for t in forbidden['tables'] if t['name']=='documents')==[]
    assert client.get(f"/projects/{project['id']}/storage?focus_table=not_a_table&focus_id=x",headers=auth).status_code==422
    foreign=register(client,email='foreign-storage@example.com')
    assert client.get(f"/projects/{project['id']}/storage",headers=foreign).status_code==404
    assert client.get(f"/projects/{project['id']}/storage?sample=100",headers=auth).status_code==422
