"""
tools/arxiv_tool.py — arXiv paper search and async PDF downloader.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import arxiv
import httpx
from loguru import logger


@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    pdf_url: str
    entry_url: str
    categories: List[str] = field(default_factory=list)
    local_pdf_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published": self.published,
            "pdf_url": self.pdf_url,
            "entry_url": self.entry_url,
            "categories": self.categories,
            "local_pdf_path": self.local_pdf_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PaperMetadata":
        fields = {
            "arxiv_id", "title", "authors", "abstract", "published",
            "pdf_url", "entry_url", "categories", "local_pdf_path"
        }
        return cls(**{k: v for k, v in d.items() if k in fields})


def search_papers(query: str, max_results: int = 5) -> List[PaperMetadata]:
    """Search arXiv and return list of PaperMetadata."""
    logger.info(f"Searching arXiv: '{query}' (max={max_results})")
    client = arxiv.Client(num_retries=3, delay_seconds=2.0)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results: List[PaperMetadata] = []
    seen_ids: set = set()

    for result in client.results(search):
        arxiv_id = result.entry_id.split("/")[-1]
        clean_id = re.sub(r"v\d+$", "", arxiv_id)
        if clean_id in seen_ids:
            continue
        seen_ids.add(clean_id)

        paper = PaperMetadata(
            arxiv_id=clean_id,
            title=result.title.strip(),
            authors=[str(a) for a in result.authors],
            abstract=result.summary.strip().replace("\n", " "),
            published=result.published.strftime("%Y-%m-%d") if result.published else "",
            pdf_url=result.pdf_url,
            entry_url=result.entry_id,
            categories=result.categories,
        )
        results.append(paper)

    logger.info(f"arXiv returned {len(results)} papers.")
    return results


async def _download_single(paper: PaperMetadata, dest_dir: Path,
                           client: httpx.AsyncClient) -> PaperMetadata:
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", paper.arxiv_id) + ".pdf"
    dest_path = dest_dir / safe_name

    if dest_path.exists() and dest_path.stat().st_size > 1024:
        paper.local_pdf_path = str(dest_path)
        return paper

    try:
        async with client.stream("GET", paper.pdf_url, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                async for chunk in resp.aiter_bytes(8192):
                    f.write(chunk)
        paper.local_pdf_path = str(dest_path)
        logger.info(f"Downloaded {safe_name} ({dest_path.stat().st_size // 1024} KB)")
    except Exception as exc:
        logger.error(f"Failed to download {paper.arxiv_id}: {exc}")

    return paper


async def download_pdfs_async(papers: List[PaperMetadata], dest_dir: Path,
                              concurrency: int = 3) -> List[PaperMetadata]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(p, c):
        async with sem:
            return await _download_single(p, dest_dir, c)

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        tasks = [_bounded(p, client) for p in papers]
        return list(await asyncio.gather(*tasks))


def download_pdfs(papers: List[PaperMetadata], dest_dir: Path,
                  concurrency: int = 3) -> List[PaperMetadata]:
    """Sync wrapper around download_pdfs_async."""
    return asyncio.run(download_pdfs_async(papers, dest_dir, concurrency))
