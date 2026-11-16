"""The built index, loaded read-only: sections (pages), search chunks, their embeddings and the abbreviation glossaries.
books.py builds every search and lookup on top of it; ingest/build_index.py writes it."""
import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dance_rag_agentic.config import EMBEDDING_MODEL_NAME, INDEX_CACHE_DIR
from dance_rag_agentic.ingest.glossary import fill_from_same_style

_embedding_model = None


def embedding_model():
    """Loaded once, on first use (loading takes a few seconds)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


@dataclass
class KnowledgeBase:
    pages: pd.DataFrame                 # merged records (bibr-first), one per table / prose section
    chunks: pd.DataFrame                # retrieval units, 'parent' -> row in pages
    embeddings: np.ndarray              # one normalised vector per chunk
    abbreviations: dict                 # {dance: {column key: {abbreviation: meaning}}}

    def __post_init__(self):
        style_per_dance = self.pages.groupby("dance")["style"].first().to_dict()
        self.abbreviations = fill_from_same_style(self.abbreviations, style_per_dance)

    # ------------------------------------------------------------------ what is in here
    @property
    def dances(self):
        return sorted(self.pages.dance.unique())

    def figures_per_dance(self):
        """{dance: [figure, ...]}"""
        charts = self.pages[self.pages.kind == "chart"]
        return {dance: sorted(group.figure.unique()) for dance, group in charts.groupby("dance")}

    @property
    def known_figures(self):
        return sorted(self.pages.figure.dropna().unique())

    def dances_with_figure(self, figure):
        """Figure names repeat across books: 'Fan' -> ['Cha Cha', 'Rumba']."""
        return sorted(self.pages[self.pages.figure == figure].dance.unique())

    def chart_pages_of(self, figure, dance):
        charts = self.pages[(self.pages.figure == figure) & (self.pages.dance == dance) & (self.pages.kind == "chart")]
        return sorted(charts.page.unique())

    # ------------------------------------------------------------------ direct lookup
    def chart_parent_ids(self, figure, roles=(), dances=()):
        """All charts (Side 1 + 2) and NOTEs of a figure, optionally limited to roles / dances."""
        selection = self.pages[self.pages.figure == figure]
        if roles:                                       # a shared chart ('Man and Lady') belongs to both roles
            shared = selection.role.fillna("").str.contains(r" (?:and|or) ", regex=True)
            selection = selection[selection.role.isin(roles) | shared]
        if dances:
            selection = selection[selection.dance.isin(dances)]
        return list(selection.sort_values(["dance", "role", "side"], na_position="last").index)


def load_knowledge_base():
    """The index from the last build (`make build`)."""
    build_info_file = INDEX_CACHE_DIR / "build_info.json"
    if not build_info_file.exists():
        raise FileNotFoundError(f"no index in {INDEX_CACHE_DIR}: run `make build` first")
    build_info = json.loads(build_info_file.read_text())
    if build_info["embedding_model"] != EMBEDDING_MODEL_NAME:
        raise ValueError(f"index was built with {build_info['embedding_model']}, config says "
                         f"{EMBEDDING_MODEL_NAME}: rebuild the index")
    return KnowledgeBase(pages=pd.read_pickle(INDEX_CACHE_DIR / "pages.pkl"),
                         chunks=pd.read_pickle(INDEX_CACHE_DIR / "chunks.pkl"),
                         embeddings=np.load(INDEX_CACHE_DIR / "embeddings.npy"),
                         abbreviations=pickle.loads((INDEX_CACHE_DIR / "abbreviations.pkl").read_bytes()))
