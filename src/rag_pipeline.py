""" RAG pipeline for question answering over customer complaint excerpts. """
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

from src.prompts import REFUSAL, build_prompt, build_yes_no_prompt
from src.retriever import dynamic_k


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    prompt: Optional[str] = None


class RAGPipeline:
    def __init__(self, retriever, generator, k: int = 3, max_context_chars: int = 900):
        self.retriever = retriever
        self.generator = generator
        self.k = k
        self.max_context_chars = max_context_chars

        # Aggregate decoding defaults (give enough room to produce 3–7 bullets)
        self.aggregate_generate_kwargs = {
            "max_new_tokens": 240,
            "num_beams": 4,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
        }

    def answer(self, question: str, k: Optional[int] = None, return_prompt: bool = False) -> RAGResult:
        q = (question or "").strip()
        if not q:
            return RAGResult(question=question, answer=REFUSAL, sources=[], prompt=None)

        # Use dynamic_k for aggregate questions unless caller overrides k explicitly
        if k is None:
            use_k = dynamic_k(q, base_k=self.k)
        else:
            use_k = k

        retrieved = self.retriever.retrieve(q, k=use_k)

        # Build context chunks (with headers) for normal QA + yes/no evidence quoting.
        context_chunks: List[str] = []
        for i, r in enumerate(retrieved, start=1):
            text = (r.get("text") or "").strip()
            if not text:
                continue

            trimmed = self._trim_text(
                q, text, max_chars=self.max_context_chars)
            if not trimmed:
                continue

            header_bits = []
            if r.get("complaint_id") is not None:
                header_bits.append(f"complaint_id={r.get('complaint_id')}")
            if r.get("product"):
                header_bits.append(f"product={r.get('product')}")
            if r.get("score") is not None:
                header_bits.append(
                    f"score={r.get('score'):.4f}"
                    if isinstance(r.get("score"), (int, float))
                    else f"score={r.get('score')}"
                )

            header = f"[{i}]"
            if header_bits:
                header += " (" + ", ".join(header_bits) + ")"

            context_chunks.append(f"{header} {trimmed}")

        # Route: Yes/No schema
        if self._is_yes_no_question(q):
            prompt = build_yes_no_prompt(context_chunks, q)
            model_text = self.generator.generate(prompt).text
            answer = self._apply_yes_no_guards(q, model_text, context_chunks)

        else:
            # Route: Aggregate bullets
            if self._is_aggregate_question(q):
                # Strip headers BEFORE prompting to prevent the model copying metadata
                aggregate_chunks = [self._strip_context_header(
                    ch) for ch in context_chunks]
                prompt = self._build_aggregate_prompt(aggregate_chunks, q)

                raw_answer = self.generator.generate(
                    prompt,
                    generate_kwargs=self.aggregate_generate_kwargs,
                ).text

                answer = self._coerce_aggregate_bullets(
                    raw_answer, aggregate_chunks, q)

            # Route: Free-form grounded answer
            else:
                prompt = build_prompt(context_chunks, q)
                answer = self.generator.generate(prompt).text
                if not answer.strip():
                    answer = REFUSAL

        return RAGResult(
            question=q,
            answer=answer,
            sources=retrieved,
            prompt=prompt if return_prompt else None,
        )

    # ----------------------------
    # Question classification
    # ----------------------------
    def _is_yes_no_question(self, question: str) -> bool:
        q = (question or "").strip().lower()
        starters = (
            "do ", "does ", "did ",
            "is ", "are ", "was ", "were ",
            "can ", "could ", "will ", "would ",
            "have ", "has ", "had ",
            "should ",
        )
        if q.startswith(starters):
            return True
        if "do complaints mention" in q or "are there complaints" in q or "do users report" in q:
            return True
        return False

    def _is_aggregate_question(self, question: str) -> bool:
        q = (question or "").strip().lower()
        triggers = (
            "most common", "common issues", "recurring", "frequent", "often",
            "top issues", "main issues", "typical issues", "patterns",
            "what are customers complaining", "what issues do customers",
            "what are the issues", "what problems do customers",
            "themes", "trends",
        )
        if any(t in q for t in triggers):
            return True
        if q.startswith("what are") and any(w in q for w in ("issues", "problems", "complaints")) and any(
            w in q for w in ("regarding", "about", "with")
        ):
            return True
        return False

    # ----------------------------
    # Header stripping (FIXED)
    # ----------------------------
    def _strip_context_header(self, chunk: str) -> str:
        """
        Remove leading chunk headers like:
          [1] (complaint_id=..., product=..., score=...) <text>
          [1] <text>
        """
        s = (chunk or "").strip()
        s = re.sub(r"^$\d+$\s*", "", s)
        s = re.sub(r"^$[^)]*$\s*", "", s)
        return s.strip()

    # ----------------------------
    # Aggregate prompt + hard coercion
    # ----------------------------
    def _build_aggregate_prompt(self, context_chunks: List[str], question: str) -> str:
        ctx = "\n\n".join(
            context_chunks) if context_chunks else "(no retrieved context)"
        return (
            "You are analyzing customer complaint excerpts.\n"
            "Task: Summarize the most common recurring issues.\n\n"
            "Rules:\n"
            "- Use ONLY the provided excerpts.\n"
            "- Output MUST start with exactly: ISSUES:\n"
            "- After that, output ONLY bullets (3–7).\n"
            "- Do NOT include bracketed markers like [1] or any IDs/scores/metadata.\n"
            "- Each bullet: 6–18 words, describing an ISSUE (not a quote).\n"
            f"- If insufficient, output exactly: {REFUSAL}\n\n"
            "EXCERPTS:\n"
            f"{ctx}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )

    def _coerce_aggregate_bullets(self, answer: str, context_chunks: List[str], question: str) -> str:
        a = (answer or "").strip()
        if a and self._aggregate_answer_is_good(a):
            return a

        retry_prompt = (
            self._build_aggregate_prompt(context_chunks, question)
            + "\n\nCRITICAL:\n"
              "- Never output '[' or ']'.\n"
              "- Never output complaint_id/doc_id/score/product.\n"
              "- Do not copy-paste excerpt text; summarize issues.\n"
        )
        retry = (self.generator.generate(
            retry_prompt,
            generate_kwargs=self.aggregate_generate_kwargs,
        ).text or "").strip()

        if retry and self._aggregate_answer_is_good(retry):
            return retry

        fallback = self._deterministic_issue_bullets(context_chunks, question)
        return fallback or REFUSAL

    def _aggregate_answer_is_good(self, text: str) -> bool:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return False

        # Allow exact refusal
        if len(lines) == 1 and lines[0] == REFUSAL:
            return True

        # Require header line
        if lines[0] != "ISSUES:":
            return False

        bullet_lines = lines[1:]
        if not bullet_lines:
            return False

        if not all(ln.startswith(("-", "*", "•")) for ln in bullet_lines):
            return False

        for ln in bullet_lines:
            low = ln.lower()

            # reject ANY metadata leakage
            if "[" in ln or "]" in ln:
                return False
            if any(m in low for m in ("complaint_id", "doc_id", "score=", "product=")):
                return False

            # Keep bullets short and non-boilerplate
            if len(ln.split()) > 24:
                return False
            if low.startswith(("- i ", "- to whom", "- submitted", "- dear")):
                return False

        return True

    def _deterministic_issue_bullets(self, context_chunks: List[str], question: str) -> str:
        if not context_chunks:
            return ""

        boilerplate_starts = (
            "to whom", "submitted", "dear", "i am writing", "this complaint", "without prejudice"
        )

        candidates: List[str] = []
        for ch in context_chunks[:8]:
            txt = (ch or "").strip()
            if not txt:
                continue

            # Split into simple sentence-ish fragments
            parts = [p.strip()
                     for p in re.split(r"[\.!\?\n]+", txt) if p.strip()]
            for p in parts[:2]:
                p_low = p.lower()
                if p_low.startswith(boilerplate_starts):
                    continue

                p = re.sub(r"\s+", " ", p)
                words = p.split()
                if len(words) < 6:
                    continue

                p_short = " ".join(words[:16])

                if any(x in p_short.lower() for x in ("[", "]", "complaint_id", "doc_id", "score=", "product=")):
                    continue

                candidates.append(p_short)

        # dedupe
        out: List[str] = []
        for c in candidates:
            if c not in out:
                out.append(c)

        out = out[:7]
        if not out:
            return ""

        bullets = "\n".join(f"- {c}" for c in out)
        return f"ISSUES:\n{bullets}"

    # ----------------------------
    # Context trimming (keyword-windowing) - your version kept
    # ----------------------------
    def _trim_text(self, question: str, text: str, max_chars: int = 900) -> str:
        t = (text or "").strip()
        if len(t) <= max_chars:
            return t

        domain_kws = self._domain_keywords_for_question(question)
        if not domain_kws:
            q = re.sub(r"[^a-z0-9\s]", " ", (question or "").lower())
            q_tokens = [w for w in q.split() if len(w) >= 4]
            stop = {
                "what", "when", "where", "which", "about", "customers",
                "complaints", "regarding", "mention", "describe"
            }
            domain_kws = [w for w in q_tokens if w not in stop][:10]

        lower = t.lower()
        hits: List[int] = []
        for kw in domain_kws:
            idx = lower.find(kw)
            if idx != -1:
                hits.append(idx)

        if not hits:
            head = t[: max_chars // 2].rstrip()
            tail = t[-(max_chars // 2):].lstrip()
            combined = (head + "\n...\n" + tail).strip()
            return combined[:max_chars]

        center = min(hits)
        start = max(0, center - max_chars // 3)
        end = min(len(t), start + max_chars)
        window = t[start:end].strip()

        if start > 0:
            cut = window.find(" ")
            if cut != -1 and cut < 30:
                window = window[cut + 1:].lstrip()

        return window[:max_chars].strip()

    # ----------------------------
    # Yes/No guards (your version, but FIX header strip inside evidence picker)
    # ----------------------------
    def _apply_yes_no_guards(self, question: str, model_text: str, context_chunks: List[str]) -> str:
        raw = (model_text or "").strip()
        if not raw:
            return f"MENTIONED: I don't know\nEVIDENCE: {REFUSAL}"

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

        if len(lines) < 2:
            mentioned = self._normalize_mentioned(
                lines[0] if lines else "") or "I don't know"
            if mentioned in {"No", "I don't know"} and self._context_indicates_mentioned(question, context_chunks):
                mentioned = "Yes"
            evidence = REFUSAL if mentioned == "I don't know" else self._pick_evidence_snippet(
                question, context_chunks)
            return f"MENTIONED: {mentioned}\nEVIDENCE: {evidence}"

        m_line, e_line = lines[0], lines[1]
        mentioned_raw = m_line.split(":", 1)[1].strip(
        ) if m_line.lower().startswith("mentioned:") else m_line.strip()
        evidence_raw = e_line.split(":", 1)[1].strip(
        ) if e_line.lower().startswith("evidence:") else e_line.strip()

        mentioned = self._normalize_mentioned(mentioned_raw) or "I don't know"

        if mentioned in {"No", "I don't know"} and self._context_indicates_mentioned(question, context_chunks):
            evidence2 = self._pick_evidence_snippet(question, context_chunks)
            return f"MENTIONED: Yes\nEVIDENCE: {evidence2}"

        if mentioned == "I don't know":
            return f"MENTIONED: I don't know\nEVIDENCE: {REFUSAL}"

        evidence = evidence_raw
        if not self._evidence_is_usable(evidence, context_chunks):
            evidence = self._pick_evidence_snippet(question, context_chunks)

        return f"MENTIONED: {mentioned}\nEVIDENCE: {evidence}"

    def _evidence_is_usable(self, evidence: str, context_chunks: List[str]) -> bool:
        ev = (evidence or "").strip()
        if not ev:
            return False
        if ev == REFUSAL:
            return True
        if len(ev.split()) > 28:
            return False
        ctx = "\n".join(context_chunks).lower()
        return ev.lower() in ctx

    def _normalize_mentioned(self, val: str) -> Optional[str]:
        v = (val or "").strip().lower().strip(".")
        if v in {"yes", "y"}:
            return "Yes"
        if v in {"no", "n"}:
            return "No"
        if v in {"i don't know", "idk", "unknown", "not sure", "cannot determine"}:
            return "I don't know"
        return None

    def _pick_evidence_snippet(self, question: str, context_chunks: List[str]) -> str:
        if not context_chunks:
            return REFUSAL

        q_low = (question or "").lower()
        domain_kws = self._domain_keywords_for_question(question)
        wants_situations = ("what situations" in q_low) or (
            "situations are described" in q_low)

        domain = self._domain_name_for_question(question)
        domain_markers: Dict[str, List[str]] = {
            "discrimination": ["denied", "refused", "because", "rac", "bias", "proof", "injured", "treated"],
            "overdraft/fees": ["fee", "fees", "charged", "overdraft", "nsf", "unexpected", "told", "promised"],
            "fraud/unauthorized": ["unauthorized", "fraud", "stolen", "not authorized", "chargeback", "dispute"],
            "credit reporting": ["credit report", "reported", "delinquent", "late", "inaccurate", "erroneous", "dispute"],
            "posting delay": ["posting", "posted", "delay", "pending", "misapplied", "applied", "took", "days"],
            "close account": ["closed", "being closed", "close", "cancel", "cancellation", "terminate", "notified"],
        }
        situation_markers = [
            "because", "when", "after", "before", "while", "during",
            "refused", "denied", "wouldn't", "won't", "would not",
            "charged", "fee", "overdraft", "late", "reported",
            "closed", "cancel", "cancelling", "cancellation",
            "unauthorized", "fraud", "stolen",
            "dispute", "error", "incorrect",
        ]
        situation_markers.extend(domain_markers.get(domain or "", []))

        def strip_header(s: str) -> str:
            s = (s or "").strip()
            s = re.sub(r"^$\d+$\s*", "", s)
            s = re.sub(r"^$[^)]*$\s*", "", s)
            return s.strip()

        def split_sentences(s: str) -> List[str]:
            s = strip_header(s)
            parts = re.split(r"(?<=[\.\!\?])\s+|\n+", s)
            return [p.strip() for p in parts if p.strip()]

        def score_text(s: str) -> int:
            s_low = s.lower()
            score = 0
            for kw in domain_kws:
                if kw in s_low:
                    score += 6
            if wants_situations:
                for m in situation_markers:
                    if m in s_low:
                        score += 1
            if domain == "discrimination" and "falls under discrimination" in s_low:
                score -= 2
            n_words = len(s.split())
            if n_words >= 8:
                score += 1
            if n_words >= 14:
                score += 1
            return score

        best_chunk = context_chunks[0]
        best_chunk_score = -1
        for ch in context_chunks:
            ch_low = ch.lower()
            chunk_score = sum(
                1 for kw in domain_kws if kw in ch_low) if domain_kws else 0
            if chunk_score > best_chunk_score:
                best_chunk_score = chunk_score
                best_chunk = ch

        candidate_sentences: List[str] = []
        chunks_to_search = [best_chunk] + \
            [c for c in context_chunks if c is not best_chunk]
        for ch in chunks_to_search:
            sents = split_sentences(ch)
            if domain_kws:
                kw_sents = [s for s in sents if any(
                    kw in s.lower() for kw in domain_kws)]
                candidate_sentences.extend(kw_sents if kw_sents else sents)
            else:
                candidate_sentences.extend(sents)

        best_sent = ""
        best_score = -1
        for s in candidate_sentences:
            sc = score_text(s)
            if sc > best_score:
                best_score = sc
                best_sent = s

        if not best_sent:
            best_sent = strip_header(best_chunk)

        words = best_sent.split()
        if len(words) <= 20:
            return " ".join(words).strip()

        if domain_kws:
            sent_low = best_sent.lower()
            idxs = [sent_low.find(kw)
                    for kw in domain_kws if sent_low.find(kw) != -1]
            if idxs:
                char_idx = min(idxs)
                word_idx = max(0, best_sent[:char_idx].count(" "))
                start = max(0, word_idx - 6)
                end = min(len(words), start + 18)
                return " ".join(words[start:end]).strip()

        return " ".join(words[:18]).strip()

    # ----------------------------
    # Keyword taxonomy + mention detection (your version)
    # ----------------------------
    def _domain_name_for_question(self, question: str) -> Optional[str]:
        q = (question or "").lower()
        for name, kws in self._domains():
            if any(k in q for k in kws):
                return name
        return None

    def _domain_keywords_for_question(self, question: str) -> List[str]:
        q = (question or "").lower()
        for _, kws in self._domains():
            if any(k in q for k in kws):
                return kws
        return []

    def _context_indicates_mentioned(self, question: str, context_chunks: List[str]) -> bool:
        q = (question or "").lower()
        ctx = "\n".join(context_chunks).lower()
        for _, kws in self._domains():
            if any(k in q for k in kws):
                return any(k in ctx for k in kws)
        q_clean = re.sub(r"[^a-z0-9\s]", " ", q)
        toks = [w for w in q_clean.split() if len(w) >= 6]
        return any(t in ctx for t in toks[:6])

    def _domains(self) -> List[Tuple[str, List[str]]]:
        return [
            ("discrimination", ["discrimination", "discriminat",
             "rac", "bias", "fair lending", "protected class"]),
            ("overdraft/fees", ["overdraft", "fee", "fees",
             "nsf", "insufficient", "unexpected fee", "charged"]),
            ("fraud/unauthorized", ["fraud", "unauthorized",
             "chargeback", "stolen", "identity theft", "not authorized"]),
            ("credit reporting", ["credit report", "credit reporting",
             "experian", "equifax", "transunion", "delinquent", "late payment"]),
            ("posting delay", ["posting", "posted", "delay", "pending",
             "misapplied", "applied to wrong", "payment not posted"]),
            ("close account", ["close account", "closing account", "closed my account", "account closed",
                               "account being closed", "being closed", "account closure", "close my account",
                               "cancel", "cancelling", "cancellation", "terminate", "termination", "won't close"]),
        ]
