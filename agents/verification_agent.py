"""
agents/verification_agent.py — Agent 3: Verify claims against retrieved evidence.
"""
from __future__ import annotations

from loguru import logger

from rag.retriever import format_context, retrieve
from tools.llm import get_llm
from workflows.states import ClaimVerdict, VerifiedClaim, WorkflowState

_SYSTEM = (
    "You are a fact-checking AI. Given a claim and supporting context from academic papers, "
    "determine whether the claim is supported by the evidence. "
    "Be concise, precise, and academically rigorous."
)

_VERIFY_PROMPT = """
You are verifying the following claim against retrieved evidence from research papers.

CLAIM:
{claim}

RETRIEVED EVIDENCE:
{context}

Respond in this exact format:
VERDICT: <VERIFIED | UNVERIFIED | CONTRADICTED>
CONFIDENCE: <0.0 to 1.0>
EXPLANATION: <one or two sentences>
"""


def _parse_verdict(response: str) -> tuple[ClaimVerdict, float, str]:
    """Parse the LLM's structured verification response."""
    verdict = ClaimVerdict.UNVERIFIED
    confidence = 0.5
    explanation = response

    for line in response.splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            v = line.replace("VERDICT:", "").strip().upper()
            if "VERIFIED" in v and "UN" not in v:
                verdict = ClaimVerdict.VERIFIED
            elif "CONTRADICTED" in v:
                verdict = ClaimVerdict.CONTRADICTED
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.replace("CONFIDENCE:", "").strip())
            except ValueError:
                pass
        elif line.startswith("EXPLANATION:"):
            explanation = line.replace("EXPLANATION:", "").strip()

    return verdict, confidence, explanation


def _extract_claims(state: WorkflowState) -> list[str]:
    """Extract verifiable claims from summaries and synthesis."""
    claims: list[str] = []
    for summary in state.summaries:
        if summary.key_contributions:
            claims.append(summary.key_contributions[:400])
        if summary.results:
            claims.append(summary.results[:400])
    if state.cross_paper_synthesis:
        # Break synthesis into sentences as individual claims
        sentences = [s.strip() for s in state.cross_paper_synthesis.split(".") if len(s.strip()) > 30]
        claims.extend(sentences[:5])
    return claims[:10]   # cap at 10 claims to control latency


def run_verification_agent(state: WorkflowState) -> WorkflowState:
    """
    LangGraph node: Verification Agent.

    Verifies key claims from summaries against FAISS-retrieved evidence.
    Tags each as VERIFIED / UNVERIFIED / CONTRADICTED.
    Tracks hallucination rate in metrics.
    """
    state.current_agent = "verification_agent"
    state.log("🔬 Verification Agent: Checking claims against evidence...")
    logger.info(f"[VerificationAgent] session={state.session_id}")

    claims = _extract_claims(state)
    if not claims:
        state.log("⚠️  No claims to verify — skipping.")
        return state

    state.log(f"  Verifying {len(claims)} claims...")
    llm = get_llm()
    verified: list[VerifiedClaim] = []
    contradicted = 0

    for i, claim in enumerate(claims):
        try:
            # Retrieve supporting evidence for this specific claim
            chunks = retrieve(
                query=claim,
                session_id=state.session_id,
                k=3,
            )
            context = format_context(chunks, max_chars=2000)

            prompt = _VERIFY_PROMPT.format(claim=claim, context=context or "No evidence found.")
            response = llm.generate(prompt, agent="verification_agent", system=_SYSTEM,
                                    temperature=0.1, max_tokens=256)
            state.metrics.total_llm_calls += 1

            verdict, confidence, explanation = _parse_verdict(response)
            if verdict == ClaimVerdict.CONTRADICTED:
                contradicted += 1

            vc = VerifiedClaim(
                claim=claim,
                verdict=verdict,
                evidence=explanation,
                confidence=confidence,
            )
            verified.append(vc)
            state.log(f"  [{verdict.value.upper()}] Claim {i+1}/{len(claims)} "
                      f"(conf={confidence:.2f})")
        except Exception as exc:
            logger.error(f"[VerificationAgent] Error on claim {i}: {exc}")
            verified.append(VerifiedClaim(
                claim=claim,
                verdict=ClaimVerdict.UNVERIFIED,
                evidence=f"Verification error: {exc}",
            ))

    state.verified_findings = verified

    # Compute hallucination proxy rate
    total = len(verified)
    if total > 0:
        unverified_count = sum(
            1 for v in verified if v.verdict != ClaimVerdict.VERIFIED
        )
        state.metrics.hallucination_rate = round(unverified_count / total, 3)

    state.log(f"✅ Verification done: {len(verified)} claims "
              f"({contradicted} contradicted, "
              f"hallucination_rate={state.metrics.hallucination_rate:.1%})")

    return state
