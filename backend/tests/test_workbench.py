"""Real saved recipes, isolated experiments and construction provenance through the API."""
import os
import pytest
from conftest import register, wait_for_job
from test_indexes import make_docs, build
from test_evaluation import dataset, seed_questions
from studio.config import settings
from studio.vectorstores import STORES


def recipe(client, auth, project, name="Baseline", store="qdrant", **extra):
    body={"name":name,"index":{"name":name,"provider":"local_hash","store":store,"chunking":{"method":"recursive","params":{"size":40,"overlap":5}}},
          "retrieval":{"query":"Which rocket carried Apollo?","mode":"hybrid","top_k":3},**extra}
    r=client.post(f"/projects/{project['id']}/pipelines",headers=auth,json=body)
    assert r.status_code==201,r.text
    return r.json()


def execute(client,auth,p):
    r=client.post(f"/pipelines/{p['id']}/run",headers=auth)
    assert r.status_code==202,r.text
    data=r.json()
    job=wait_for_job(client,auth,data['job']['id'],timeout=90)
    assert job['status']=='succeeded',job
    return client.get(f"/workbench-runs/{data['id']}",headers=auth).json()


@pytest.fixture()
def docs(client,auth,project):
    return make_docs(client,auth,project)


@pytest.mark.parametrize("store",list(STORES))
def test_pipeline_executes_each_real_store(client,auth,project,docs,monkeypatch,store):
    from studio.vectorstores import chroma_store,lancedb_store,pgvector_store
    monkeypatch.setattr(chroma_store,"_client",None)
    monkeypatch.setattr(lancedb_store,"_db",None)
    monkeypatch.setattr(lancedb_store,"_meta",{})
    monkeypatch.setattr(pgvector_store,"_engine",None)
    monkeypatch.setattr(settings,"pgvector_url",os.getenv("TEST_PGVECTOR_URL","postgresql://postgres:studio@localhost:5434/vec"))
    if not STORES[store].status()[0]:pytest.skip("store unavailable")
    p=recipe(client,auth,project,store=store)
    result=execute(client,auth,p)
    assert result['status']=='done' and [s['stage'] for s in result['artifacts']]==['index','retrieve']
    case=result['result']['cases'][0]
    assert 'Saturn V' in case['retrieval']['results'][0]['text']
    assert all(a['status']=='done' for a in result['artifacts'])
    proof=client.get(f"/indexes/{case['index_id']}/construction",headers=auth).json()
    assert proof['store']==store and proof['samples'][0]['vector_preview']
    assert proof['samples'][0]['embedding_input']
    assert client.get(f"/workbench-runs/{result['id']}/export",headers=auth).status_code==200


def test_recipe_versions_are_immutable_and_reuse_keeps_index(client,auth,project,docs):
    p1=recipe(client,auth,project)
    p2=recipe(client,auth,project,retrieval={"query":"ZX-4471","mode":"tfidf"})
    assert p1['version']==1 and p2['version']==2
    assert p1['config']['retrieval']['query']!='ZX-4471'
    idx=build(client,auth,project)
    p3=recipe(client,auth,project,name="Reuse",reuse_index_id=idx['id'])
    result=execute(client,auth,p3)
    assert result['result']['cases'][0]['index_id']==idx['id']
    assert result['artifacts'][0]['detail']['reuse'] is True


def test_experiment_freezes_questions_and_builds_distinct_indexes(client,auth,project,docs):
    p1=recipe(client,auth,project)
    p2=recipe(client,auth,project,name="Words",retrieval={"query":"ZX-4471","mode":"tfidf","top_k":3})
    ds=dataset(client,auth,project);seed_questions(client,auth,ds)
    r=client.post(f"/projects/{project['id']}/experiments",headers=auth,json={"name":"Controlled comparison","pipeline_ids":[p1['id'],p2['id']],"dataset_id":ds['id']})
    assert r.status_code==202,r.text
    exp=r.json()
    assert exp['config']['dataset_id']!=ds['id'] and exp['config']['questions']==4
    assert wait_for_job(client,auth,exp['job']['id'],timeout=90)['status']=='succeeded'
    result=client.get(f"/workbench-runs/{exp['id']}",headers=auth).json()
    cases=result['result']['cases']
    assert len({c['index_id'] for c in cases})==2
    assert all(c['metrics']['failed_questions']==0 for c in cases)
    assert all(c['metrics']['questions']==4 for c in cases)
    assert len(cases[0]['per_question'])==4


def test_pipeline_types_authorization_and_invalid_experiment(client,auth,project,docs):
    p=recipe(client,auth,project)
    other=register(client,email='other-workbench@example.com')
    assert client.post(f"/pipelines/{p['id']}/run",headers=other).status_code==404
    bad=client.post(f"/projects/{project['id']}/pipelines",headers=auth,json={"name":"bad","index":{"name":"bad","provider":"local_hash","chunking":{"method":"recursive"}},"retrieval":{"query":"x","expand":"parent"}})
    assert bad.status_code==422 and 'parent' in bad.text.lower()
    assert client.get(f"/projects/{project['id']}/pipelines",headers=other).status_code==404
    result=execute(client,auth,p)
    assert client.get(f"/workbench-runs/{result['id']}/export",headers=other).status_code==404


def test_cancelled_pipeline_preserves_artifacts_without_stuck_index(client,auth,project,docs,monkeypatch):
    from studio.jobs import index_job
    def interrupted(ctx):
        raise InterruptedError("test cancellation during build")
    monkeypatch.setattr(index_job,"build_index",interrupted)
    p=recipe(client,auth,project)
    r=client.post(f"/pipelines/{p['id']}/run",headers=auth).json()
    assert wait_for_job(client,auth,r['job']['id'])['status']=='cancelled'
    out=client.get(f"/workbench-runs/{r['id']}",headers=auth).json()
    assert out['status']=='cancelled' and out['artifacts'][0]['status']=='failed'
    idx=client.get(f"/indexes/{out['result']['cases'][0]['index_id']}",headers=auth).json()
    assert idx['status']=='failed'
