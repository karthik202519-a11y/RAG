"""Query the stored knowledge graph: local search around the entities a question mentions, and a global overview."""
import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import GraphEdge, GraphEntity, IndexChunk
from .extract import norm_name


def _graph(db: Session, index_id: str) -> tuple[nx.Graph, dict[str, GraphEntity]]:
    ents = {e.id: e for e in db.scalars(select(GraphEntity).where(GraphEntity.index_id == index_id))}
    g = nx.Graph()
    g.add_nodes_from(ents)
    for e in db.scalars(select(GraphEdge).where(GraphEdge.index_id == index_id)):
        if g.has_edge(e.src_id, e.dst_id):
            g[e.src_id][e.dst_id]["weight"] += 1
        else:
            g.add_edge(e.src_id, e.dst_id, weight=1)
    return g, ents


def graph_overview(db: Session, index_id: str, max_nodes: int = 80) -> dict:
    g, ents = _graph(db, index_id)
    if not ents:
        return {"entities": 0, "relations": 0, "communities": [], "nodes": [], "edges": []}
    comms = [sorted(c, key=lambda n: -g.degree(n)) for c in nx.algorithms.community.greedy_modularity_communities(g)] if g.number_of_edges() else []
    comm_of = {n: i for i, c in enumerate(comms) for n in c}
    connected = [n for n in ents if g.degree(n) > 0] or list(ents)  # lone entities say nothing about structure
    top = sorted(connected, key=lambda n: (-g.degree(n), -ents[n].mentions))[:max_nodes]
    keep = set(top)
    edges = db.scalars(select(GraphEdge).where(GraphEdge.index_id == index_id)).all()
    return {
        "entities": len(ents), "relations": len(edges), "isolated": sum(1 for n in g if g.degree(n) == 0),
        "communities": [{"id": i, "size": len(c), "top": [ents[n].name for n in c[:6]]} for i, c in enumerate(comms[:12])],
        "nodes": [{"id": n, "name": ents[n].name, "type": ents[n].etype, "degree": g.degree(n), "mentions": ents[n].mentions,
                   "community": comm_of.get(n, -1)} for n in top],
        "edges": [{"source": e.src_id, "target": e.dst_id, "relation": e.relation} for e in edges if e.src_id in keep and e.dst_id in keep][:400],
    }


def graph_search(db: Session, index_id: str, query: str, top_k: int = 5, hops: int = 1) -> dict:
    """Find entities named in the question, walk `hops` steps out, and rank the chunks the walked relations came from."""
    g, ents = _graph(db, index_id)
    q = " " + norm_name(query) + " "
    seeds = [n for n, e in ents.items() if f" {e.norm} " in q]
    if not seeds:
        return {"seeds": [], "results": [], "edges": [], "note": "No known entity is named in the question, so graph search has nothing to start from. Try dense or hybrid retrieval."}
    dist: dict[str, int] = {s: 0 for s in seeds}
    frontier = set(seeds)
    for h in range(1, hops + 1):
        nxt = set()
        for n in frontier:
            for m in g.neighbors(n):
                if m not in dist:
                    dist[m] = h
                    nxt.add(m)
        frontier = nxt
    edges = db.scalars(select(GraphEdge).where(GraphEdge.index_id == index_id,
                                               GraphEdge.src_id.in_(dist) | GraphEdge.dst_id.in_(dist))).all()
    score: dict[str, float] = {}
    used = []
    for e in edges:
        d = min(dist.get(e.src_id, 99), dist.get(e.dst_id, 99))
        w = 1.0 / (1 + d) + (0.5 if e.src_id in seeds and e.dst_id in seeds else 0)  # both endpoints named in the question is strongest
        score[e.chunk_id] = score.get(e.chunk_id, 0.0) + w
        used.append(e)
    ranked = sorted(score, key=lambda c: -score[c])[:top_k]
    chunks = {c.id: c for c in db.scalars(select(IndexChunk).where(IndexChunk.id.in_(ranked)))}
    results = []
    for rank, cid in enumerate(ranked, 1):
        c = chunks.get(cid)
        if c is None:
            continue
        results.append({"rank": rank, "chunk_id": cid, "score": round(score[cid], 3), "filename": c.filename, "pages": c.pages,
                        "section": c.section, "text": c.text, "document_id": c.document_id,
                        "via": [{"source": ents[e.src_id].name, "relation": e.relation, "target": ents[e.dst_id].name} for e in used if e.chunk_id == cid][:6]})
    return {"seeds": [{"id": s, "name": ents[s].name} for s in seeds],
            "results": results,
            "edges": [{"source": ents[e.src_id].name, "relation": e.relation, "target": ents[e.dst_id].name} for e in used[:60]],
            "note": None}
