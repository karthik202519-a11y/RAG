"""Read-only, tenant-scoped application schema samples and native vector storage evidence."""
import json
import uuid
from datetime import datetime
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_, text, inspect
from sqlalchemy.orm import Session
from ..db import Base, get_db
from ..models import Project, Document, DocumentVersion, Index, IndexChunk, EvalDataset, Collection, User
from ..security import current_user, get_project
from ..embeddings.service import cache_key
from ..jobs.index_job import embed_input
from ..vectorstores import store_for
from .indexes import load_index

router=APIRouter(tags=["storage-explorer"])
SECRET={"password_hash","password","api_key","secret_key","access_token","refresh_token","authorization"}
PURPOSES={
 "users":("Account identity","Separate account identity from project content; password hashes are never displayed."),
 "workspaces":("Workspace boundary","Groups projects and memberships under one permission boundary."),
 "workspace_members":("Workspace roles","Links an account to a workspace and its role; only your membership is sampled here."),
 "projects":("Project ownership","Scopes documents, indexes, datasets, recipes and runs."),
 "collections":("Document grouping","Organizes project documents without copying their source files."),
 "documents":("Stable document identity","Keeps filename and ownership stable while content versions change."),
 "document_versions":("Immutable source and extraction versions","Stores source/extraction artifact keys and parse metadata; large files live in artifact storage."),
 "jobs":("Durable execution state","Tracks progress, retries and cancellation independently of a browser session."),
 "embedding_cache":("Reusable normalized embeddings","Content-addressed vectors avoid repeated provider calls; this shared cache has no direct chunk foreign key."),
 "indexes":("Frozen index recipe and vector namespace","Records provider, model, dimension, store and collection once, rather than repeating them in every vector payload."),
 "index_chunks":("Searchable text and provenance","Links text, page references, document version and index; its ID is also used by the native vector record."),
 "index_parents":("Larger context passages","Lets small child chunks match precisely while a parent supplies broader context; parent text is not separately embedded."),
 "retrieval_runs":("Search trace metadata","Stores the question, configuration, result IDs and timings; result IDs are JSON references rather than SQL foreign keys."),
 "eval_datasets":("Benchmark identity","Groups labelled questions so configurations can be compared on the same evidence."),
 "eval_questions":("Labelled evaluation questions","Stores evidence pages, expected keywords and splits separately from measured runs."),
 "eval_runs":("Frozen evaluation results","Keeps metrics and per-question output; index identity is a historical string reference so deleting an index does not erase results."),
 "graph_builds":("Graph extraction state","Tracks the model, sampled chunks, rejected proposals and extraction progress."),
 "graph_entities":("Normalized named entities","Deduplicates entity identity within an index so many relations can share the same node."),
 "graph_edges":("Relations and source provenance","Links source/target entities with a relation and source chunk reference; the relation remains a model proposal."),
 "audit_events":("Account/project action trail","Records who performed an action; samples are limited to this project's resource IDs and your account."),
 "pipeline_recipes":("Immutable pipeline versions","Preserves configuration choices so executions remain reproducible."),
 "workbench_runs":("Pipeline and experiment executions","Stores frozen configurations, stage artifacts and result references separately from recipes."),
}
LOGICAL=[
 {"source_table":"index_chunks","source_column":"document_id","target_table":"documents","target_column":"id","kind":"logical","why":"Stable source identity resolves the filename and project. The chunk snapshot keeps this string without a SQL foreign key."},
 {"source_table":"index_chunks","source_column":"version_id","target_table":"document_versions","target_column":"id","kind":"logical","why":"Pins source provenance to the immutable version used to build this index."},
 {"source_table":"index_chunks","source_column":"parent_id","target_table":"index_parents","target_column":"id","kind":"logical","why":"For parent/child chunking, a small searchable chunk can expand to a larger context passage."},
 {"source_table":"index_chunks","source_column":"id","target_table":"vector record","target_column":"id","kind":"logical","why":"The same chunk ID links application text to the native vector. This is not a cross-engine SQL foreign key."},
 {"source_table":"embedding_cache","source_column":"key","target_table":"index_chunks","target_column":"embedding input","kind":"logical","why":"SHA-256 of provider, model and prepared text identifies a reusable embedding. Identical input can be shared across chunks."},
 {"source_table":"graph_edges","source_column":"chunk_id","target_table":"index_chunks","target_column":"id","kind":"logical","why":"A stored string points to relation provenance; it is not declared as a SQL foreign key."},
 {"source_table":"eval_runs","source_column":"index_id","target_table":"indexes","target_column":"id","kind":"logical","why":"Historical results survive index deletion."},
 {"source_table":"retrieval_runs","source_column":"result_ids[]","target_table":"index_chunks","target_column":"id","kind":"logical","why":"An ordered JSON array preserves the ranking of returned chunks."},
 {"source_table":"document_versions","source_column":"source_key / extraction_key","target_table":"artifact storage","target_column":"object key","kind":"logical","why":"The database holds references while source files and extraction JSON live outside the relational tables."},
]
LAYOUTS={
 "qdrant":{"title":"Collection of points","objects":"A collection contains point IDs, vectors and payload metadata.","benefit":"Payload filters and a vector-search API; embedded mode is local exact search, while the server supports HNSW and query planning.","search":"The query vector searches points; point IDs retrieve the source chunks from the application database."},
 "faiss":{"title":"Binary vector index plus JSON metadata","objects":"FAISS is a search library, not a relational database. A .faiss file stores vector rows; a .json companion maps row positions to chunk IDs and payloads.","benefit":"Exact flat search is a simple baseline with no server. The full vector index is kept in memory.","search":"Search returns row positions, then the companion metadata resolves positions to chunk IDs."},
 "chroma":{"title":"Persistent collection","objects":"A collection stores IDs, embeddings and metadata. Chroma manages its own SQLite metadata and HNSW files; those are engine internals, not application tables.","benefit":"Embedded persistence and metadata filtering behind one collection API.","search":"A query embedding retrieves IDs and metadata from the collection; the application loads source text separately."},
 "lancedb":{"title":"Arrow-schema table in Lance files","objects":"The table has id, document_id, ctype and a fixed-size float32 vector column. This adapter uses an exact scan and prefilters.","benefit":"Columnar storage keeps typed metadata and vectors together on disk.","search":"Search the vector column, filter document/type columns, and return the matching row IDs."},
 "pgvector":{"title":"PostgreSQL vector table","objects":"Each index has a vs_<collection> SQL table with id, document_id, ctype and embedding vector(dimension). The adapter creates HNSW up to 2,000 dimensions; larger vectors use exact scanning.","benefit":"SQL metadata and vectors share one table. In this app the vector service can be separate from the application database.","search":"SQL orders by vector distance. A chunk ID is a logical cross-database reference, not an enforced foreign key to index_chunks."},
}


def safe(value,key="",depth=0):
    if key.lower() in SECRET:return "<redacted>"
    if isinstance(value,bytes):
        return {"bytes":len(value),"format":"float32" if key=="vector" else "binary", "first_values":[round(float(v),6) for v in np.frombuffer(value,dtype="<f4")[:8]] if key=="vector" and len(value)%4==0 else []}
    if isinstance(value,datetime):return value.isoformat()
    if isinstance(value,str):return value[:360]+(" ... [sample truncated]" if len(value)>360 else "")
    if depth>=3 and isinstance(value,(dict,list)):return "[nested data omitted from sample]"
    if isinstance(value,dict):return {str(k):safe(v,str(k),depth+1) for k,v in list(value.items())[:20]}
    if isinstance(value,list):return [safe(v,"",depth+1) for v in value[:8]]
    return value


@router.get("/projects/{project_id}/storage")
def application_storage(project_id:str, sample:int=Query(3,ge=1,le=5),focus_table:str|None=Query(None,max_length=80),focus_id:str|None=Query(None,max_length=64),user:User=Depends(current_user),db:Session=Depends(get_db)):
    project=get_project(db,user,project_id)
    tables=Base.metadata.tables
    if focus_table is not None and (focus_table not in tables or "id" not in tables[focus_table].c):
        raise HTTPException(422,"Focus a supported application table with an id column.")
    projects=select(Project.id).where(Project.id==project_id)
    docs=select(Document.id).where(Document.project_id==project_id)
    versions=select(DocumentVersion.id).where(DocumentVersion.document_id.in_(docs))
    indexes=select(Index.id).where(Index.project_id==project_id)
    datasets=select(EvalDataset.id).where(EvalDataset.project_id==project_id)
    collections=select(Collection.id).where(Collection.project_id==project_id)
    indices={i.id:i for i in db.scalars(select(Index).where(Index.project_id==project_id))}
    cache_keys=[]
    for c in db.scalars(select(IndexChunk).where(IndexChunk.index_id.in_(indexes)).order_by(IndexChunk.id).limit(200)):
        i=indices[c.index_id]
        prepared=embed_input(c.text,c.ctype,c.section,i.config.get("section_prefix",True),c.context)
        cache_keys.append(cache_key(i.provider,i.model,prepared))
    resource_ids={project_id}
    for table in tables.values():
        if "project_id" in table.c and "id" in table.c:
            resource_ids.update(db.scalars(select(table.c.id).where(table.c.project_id==project_id)))
    result=[];relationships=[]
    for name,table in tables.items():
        c=table.c;scope=None;note="Rows visible within the selected project."
        if name=="users":scope=c.id==user.id;note="Only your account; password hashes are redacted."
        elif name=="projects":scope=c.id==project_id
        elif name=="workspaces":scope=c.id==project.workspace_id
        elif name=="workspace_members":scope=and_(c.workspace_id==project.workspace_id,c.user_id==user.id)
        elif name=="audit_events":scope=and_(c.user_id==user.id,c.workspace_id==project.workspace_id,c.target.in_(resource_ids))
        elif name=="embedding_cache":scope=c.key.in_(cache_keys or [""]);note="Cache keys referenced by the first 200 project chunks; shared cached-query rows are excluded. This is not a total of the global cache."
        elif "project_id" in c:scope=c.project_id==project_id
        elif "dataset_id" in c:scope=c.dataset_id.in_(datasets)
        elif "index_id" in c:scope=c.index_id.in_(indexes)
        elif "document_id" in c:scope=c.document_id.in_(docs)
        elif "version_id" in c:scope=c.version_id.in_(versions)
        elif "collection_id" in c:scope=c.collection_id.in_(collections)
        else:note="Schema is shown; samples are withheld because this table has no supported project scope."
        columns=[]
        for col in c:
            fks=[{"table":fk.column.table.name,"column":fk.column.name} for fk in col.foreign_keys]
            columns.append({"name":col.name,"type":str(col.type),"primary_key":col.primary_key,"nullable":col.nullable,"references":fks,"redacted":col.name in SECRET})
            for fk in fks:relationships.append({"source_table":name,"source_column":col.name,"target_table":fk["table"],"target_column":fk["column"],"kind":"foreign_key","why":"An enforced SQL foreign key links these records."})
        rows=[];count=None
        if scope is not None:
            count=db.scalar(select(func.count()).select_from(table).where(scope))
            order=list(table.primary_key.columns)
            row_scope=and_(scope,c.id==focus_id) if focus_table==name and focus_id and "id" in c else scope
            rows=[{k:safe(v,k) for k,v in row.items()} for row in db.execute(select(table).where(row_scope).order_by(*order).limit(sample)).mappings()]
            if focus_table==name and focus_id:note += " Samples are focused on the requested row ID; the count still covers the original project scope."
        purpose,benefit=PURPOSES.get(name,("Application metadata","Read the declared columns and relationships to understand its role."))
        result.append({"name":name,"purpose":purpose,"benefit":benefit,"scope":note,"rows_in_scope":count,"columns":columns,"samples":rows})
    revision=db.execute(text("SELECT version_num FROM alembic_version")).scalar()
    result.append({"name":"alembic_version","purpose":"Schema migration revision","benefit":"Records which forward schema migration has been applied.","scope":"Global schema metadata only; contains no user or project content.","rows_in_scope":1,"columns":[{"name":"version_num","type":"VARCHAR(32)","primary_key":True,"nullable":False,"references":[],"redacted":False}],"samples":[{"version_num":revision}]})
    inspector=inspect(db.bind)
    for name in set(inspector.get_table_names())-set(tables)-{"alembic_version"}:
        cols=inspector.get_columns(name)
        result.append({"name":name,"purpose":"Unmapped database table","benefit":"Actual database schema is visible, but this explorer has no supported authorization scope for its contents.","scope":"Schema only; sample data is withheld.","rows_in_scope":None,"samples":[],"columns":[{"name":c["name"],"type":str(c["type"]),"primary_key":False,"nullable":c.get("nullable",True),"references":[],"redacted":c["name"] in SECRET} for c in cols]})
    return {"database_kind":db.bind.dialect.name,"schema_revision":revision,"tables":result,"relationships":relationships+LOGICAL,"sample_limit":sample,
        "layers":["Application SQL stores identity, text, provenance, recipes and evaluations.","Vector engines store vectors and filterable metadata; the same chunk ID joins the layers.","Artifact storage holds original files, extraction JSON and page images."],"engine_layouts":LAYOUTS}


def vector_summary(vector):
    v=np.asarray(vector,dtype=np.float32)
    return {"dimension":int(v.size),"dtype":"float32","norm":round(float(np.linalg.norm(v)),6),"first_values":[round(float(x),6) for x in v[:12]],"bytes_float32":int(v.size*4)}


@router.get("/indexes/{index_id}/storage")
def native_storage(index_id:str,sample:int=Query(3,ge=1,le=5),user:User=Depends(current_user),db:Session=Depends(get_db)):
    idx=load_index(db,user,index_id)
    if idx.status!="ready":raise HTTPException(409,"Inspect vector records after the index is ready.")
    vs=store_for(idx.store);records=[];objects=[];native_schema=[];indexes=[]
    try:
        if idx.store=="qdrant":
            points,_=vs.client().scroll(collection_name=idx.collection,limit=sample,with_payload=True,with_vectors=True)
            records=[{"id":uuid.UUID(str(p.id)).hex,"metadata":p.payload or {},"vector":vector_summary(p.vector)} for p in points]
            native_schema=[{"name":"id","type":"UUID (chunk ID)"},{"name":"vector","type":f"float32[{idx.dimension}]"},{"name":"payload","type":"JSON metadata"}]
            info=vs.client().get_collection(idx.collection)
            objects=[{"kind":"collection","name":idx.collection,"vector_configuration":safe(info.config.params.vectors.model_dump(mode="json"))}]
        elif idx.store=="faiss":
            c=vs._load(idx.collection)
            if c is None:raise RuntimeError("FAISS index files are missing")
            records=[{"id":c.ids[n],"metadata":c.meta[n],"vector":vector_summary(c.index.reconstruct(n))} for n in range(min(sample,len(c.ids)))]
            paths=vs._paths(idx.collection)
            objects=[{"kind":"vector index file","name":paths[0].name,"exists":paths[0].exists()},{"kind":"ID/payload companion file","name":paths[1].name,"exists":paths[1].exists()}]
            native_schema=[{"name":"row position","type":"integer"},{"name":"vector row","type":f"float32[{c.dim}]"},{"name":"ids[position]","type":"chunk ID string"},{"name":"meta[position]","type":"JSON payload"}]
        elif idx.store=="chroma":
            col=vs._coll(idx.collection);data=col.get(limit=sample,include=["embeddings","metadatas"])
            records=[{"id":cid,"metadata":data["metadatas"][n] or {},"vector":vector_summary(data["embeddings"][n])} for n,cid in enumerate(data["ids"])]
            objects=[{"kind":"collection","name":col.name,"metadata":safe(col.metadata)}]
            native_schema=[{"name":"id","type":"string"},{"name":"embedding","type":f"float32[{idx.dimension}]"},{"name":"metadata","type":"typed key/value payload"}]
        elif idx.store=="lancedb":
            table=vs._table(idx.collection)
            records=[{"id":r["id"],"metadata":{k:v for k,v in r.items() if k not in ("id","vector")},"vector":vector_summary(r["vector"])} for r in table.search().limit(sample).to_list()]
            native_schema=[{"name":field.name,"type":str(field.type)} for field in table.schema]
            objects=[{"kind":"Lance table","name":idx.collection}]
        elif idx.store=="pgvector":
            table=vs._t(idx.collection)
            with vs.engine().connect() as conn:
                rows=conn.execute(text(f"SELECT id,document_id,ctype,embedding::text AS vector FROM {table} LIMIT :n"),{"n":sample}).mappings().all()
                records=[{"id":r["id"].strip(),"metadata":{"document_id":r["document_id"],"ctype":r["ctype"]},"vector":vector_summary(json.loads(r["vector"]))} for r in rows]
                native_schema=[{"name":r[0],"type":r[1]} for r in conn.execute(text("SELECT attname,format_type(atttypid,atttypmod) FROM pg_attribute WHERE attrelid=to_regclass(:t) AND attnum>0 AND NOT attisdropped"),{"t":table})]
                indexes=[{"name":r[0],"definition":r[1]} for r in conn.execute(text("SELECT indexname,indexdef FROM pg_indexes WHERE tablename=:t"),{"t":table})]
            objects=[{"kind":"PostgreSQL vector table","name":table}]
        else:raise RuntimeError("No native storage inspection adapter")
        for r in records:
            chunk=db.get(IndexChunk,r["id"])
            r["metadata"]=safe(r["metadata"])
            r["linked_chunk"]={"table":"index_chunks","id":chunk.id,"index_id":chunk.index_id,"document_id":chunk.document_id,"version_id":chunk.version_id,"filename":chunk.filename,"pages":chunk.pages,"text":chunk.text[:700]} if chunk and chunk.index_id==idx.id else None
        return {"index_id":idx.id,"name":idx.name,"store":idx.store,"provider":idx.provider,"model":idx.model,"dimension":idx.dimension,"metric":idx.metric,"namespace":idx.collection,
                "stored_vectors":vs.count(idx.collection),"layout":LAYOUTS[idx.store],"objects":objects,"native_schema":native_schema,"native_indexes":indexes,"samples":records,
                "note":"These records are read from the native engine, not reconstructed from application metadata. Vector previews show the first 12 float32 coordinates. Native engine internals are not editable here."}
    except Exception as e:
        raise HTTPException(409,f"Could not inspect this index's native store ({type(e).__name__}). Check the store connection and index state.") from None
