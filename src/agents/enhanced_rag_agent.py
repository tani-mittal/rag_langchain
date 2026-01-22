import json
import time
import uuid
import re
import hashlib
from typing import Dict, List, Any, Iterator, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import boto3
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage
from src.knowledge_base.bedrock_kb import BedrockKnowledgeBase
import config
import logging
from collections import defaultdict


# -------------------------
# Dataclasses
# -------------------------

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime
    sources: List[str] = None
    confidence_score: float = None
    retrieval_score: float = None
    response_time: float = None
    session_id: str = None
    token_count: int = None
    retrieval_context: List[Dict] = None


@dataclass
class AccuracyMetrics:
    total_queries: int = 0
    avg_confidence: float = 0.0
    avg_retrieval_score: float = 0.0
    avg_response_time: float = 0.0
    source_coverage: float = 0.0
    user_feedback_positive: int = 0
    user_feedback_negative: int = 0
    avg_token_count: float = 0.0
    successful_retrievals: int = 0
    failed_retrievals: int = 0
    context_relevance_score: float = 0.0


@dataclass
class RetrievalResult:
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    chunk_id: str


# -------------------------
# Enhanced RAG Agent
# -------------------------

class EnhancedRAGAgent:
    def __init__(self, knowledge_base_id: str, memory_window: int = 10):
        self.kb_id = knowledge_base_id
        self.kb_manager = BedrockKnowledgeBase()
        self.session_id = str(uuid.uuid4())

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            return_messages=True,
            memory_key="chat_history"
        )

        # Bedrock runtime client (streaming)
        self.bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        )

        # Analytics / tracking
        self.accuracy_metrics = AccuracyMetrics()
        self.chat_history: List[ChatMessage] = []
        self.query_patterns = defaultdict(int)
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.performance_history = []

        # Config knobs
        self.context_window_tokens = 4000
        self.max_retrieval_attempts = 3
        self.confidence_threshold = getattr(config, "CONFIDENCE_THRESHOLD", 0.7)

        self.enable_caching = True
        self.enable_query_expansion = True

        self.logger.info(f"EnhancedRAGAgent initialized for KB: {knowledge_base_id}")
        self.logger.info(f"Session ID: {self.session_id}")

    # -------------------------
    # Main streaming API
    # -------------------------

    def stream_response(
        self,
        query: str,
        include_sources: bool = True,
        use_cache: bool = True
    ) -> Iterator[Dict[str, Any]]:

        start_time = time.time()

        try:
            # Split retrieval query vs generation query
            retrieval_query, generation_query = self._preprocess_query(query)

            cache_key = self._generate_cache_key(retrieval_query)

            # Cache hit
            if use_cache and self.enable_caching and cache_key in self.response_cache:
                yield {"type": "status", "content": "💾 Found cached response..."}
                cached = self.response_cache[cache_key]

                # Stream cached response
                for char in cached["content"]:
                    yield {"type": "content", "content": char}
                    time.sleep(0.005)

                yield {
                    "type": "complete",
                    "content": cached["content"],
                    "sources": cached.get("sources", []) if include_sources else [],
                    "metrics": cached.get("metrics", {}),
                }
                return

            yield {"type": "status", "content": "🔍 Searching knowledge base..."}

            retrieval_results = self._enhanced_retrieve(retrieval_query)

            if not retrieval_results:
                yield {
                    "type": "error",
                    "content": "No relevant documents found. Please try rephrasing your question."
                }
                return

            retrieval_score = self._calculate_retrieval_score(
                [{"score": r.score} for r in retrieval_results]
            )

            yield {"type": "status", "content": f"📚 Found {len(retrieval_results)} relevant chunks..."}

            # Context creation for LLM (includes conversation memory)
            context = self._prepare_enhanced_context(generation_query, retrieval_results)

            yield {"type": "status", "content": "🧠 Generating response..."}

            # Stream generation
            sources = []
            full_response = ""
            confidence_scores: List[float] = []

            token_estimate = 0  # approximate token count

            for chunk in self._stream_generate_enhanced(context, generation_query, retrieval_results):
                if chunk["type"] == "content":
                    full_response += chunk["content"]
                    token_estimate += max(1, len(chunk["content"]) // 4)
                    yield chunk

                elif chunk["type"] == "sources":
                    sources = chunk["sources"]

                elif chunk["type"] == "confidence":
                    confidence_scores.append(chunk["score"])

                elif chunk["type"] == "error":
                    yield chunk
                    return

            response_time = time.time() - start_time
            avg_confidence = (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores else 0.0
            )

            context_relevance = self._calculate_context_relevance(full_response, retrieval_results)

            # Memory
            self._update_memory(query, full_response)
            self._track_query_patterns(retrieval_query)

            # Metrics
            self._update_enhanced_metrics(
                confidence_score=avg_confidence,
                retrieval_score=retrieval_score,
                response_time=response_time,
                sources_count=len(sources),
                token_count=token_estimate,
                context_relevance=context_relevance,
                retrieval_success=True
            )

            # Cache if good
            if self.enable_caching and avg_confidence >= self.confidence_threshold:
                self._cache_response(
                    cache_key,
                    full_response,
                    sources,
                    metrics={
                        "confidence_score": avg_confidence,
                        "retrieval_score": retrieval_score,
                        "response_time": response_time,
                        "sources_count": len(sources),
                        "token_count": token_estimate,
                        "context_relevance": context_relevance,
                        "cache_hit": False
                    }
                )

            # Save chat history
            self.chat_history.append(ChatMessage(
                role="assistant",
                content=full_response,
                timestamp=datetime.now(),
                sources=sources,
                confidence_score=avg_confidence,
                retrieval_score=retrieval_score,
                response_time=response_time,
                session_id=self.session_id,
                token_count=token_estimate,
                retrieval_context=[r.__dict__ for r in retrieval_results]
            ))

            # Complete
            yield {
                "type": "complete",
                "content": full_response,
                "sources": sources if include_sources else [],
                "metrics": {
                    "confidence_score": avg_confidence,
                    "retrieval_score": retrieval_score,
                    "response_time": response_time,
                    "sources_count": len(sources),
                    "token_count": token_estimate,
                    "context_relevance": context_relevance,
                    "cache_hit": False
                }
            }

        except Exception as e:
            self.logger.error(f"Error in stream_response: {e}", exc_info=True)
            self._update_enhanced_metrics(retrieval_success=False)
            yield {"type": "error", "content": f"Error generating response: {str(e)}"}

    # -------------------------
    # Query preprocessing
    # -------------------------

    def _preprocess_query(self, query: str) -> tuple[str, str]:
        """
        Returns:
            retrieval_query: used ONLY for KB retrieval (keep it clean)
            generation_query: used for prompt (can include conversation context)
        """
        base_query = query.strip()

        retrieval_query = base_query
        if self.enable_query_expansion:
            retrieval_query = self._expand_query(retrieval_query)

        # Generation query may include recent context (but DO NOT use it in retrieval)
        recent_context = self._get_recent_context()

        generation_query = base_query
        if recent_context:
            generation_query = f"{base_query}\n\n(Conversation context: {recent_context})"

        return retrieval_query, generation_query

    def _expand_query(self, query: str) -> str:
        expansions = {
            "show": ["display", "present", "demonstrate"],
            "find": ["locate", "search", "identify"],
            "compare": ["contrast", "analyze", "evaluate"],
            "summarize": ["overview", "summary", "brief"],
            "explain": ["describe", "clarify", "elaborate"],
        }

        expanded_terms = []
        words = query.lower().split()

        for word in words:
            if word in expansions:
                expanded_terms.extend(expansions[word][:2])

        return f"{query} ({' '.join(expanded_terms)})" if expanded_terms else query

    def _get_recent_context(self) -> str:
        if len(self.chat_history) < 2:
            return ""

        recent_messages = self.chat_history[-2:]
        context_parts = []

        for msg in recent_messages:
            if msg.role == "user":
                context_parts.append(f"Prev Q: {msg.content[:120]}")
            elif msg.role == "assistant":
                context_parts.append(f"Prev A: {msg.content[:120]}")

        return " | ".join(context_parts)

    # -------------------------
    # Retrieval
    # -------------------------

    def _enhanced_retrieve(self, query: str) -> List[RetrievalResult]:
        all_results: List[RetrievalResult] = []

        for attempt in range(self.max_retrieval_attempts):
            try:
                search_query = query
                if attempt == 1:
                    search_query = self._make_query_specific(query)
                elif attempt == 2:
                    search_query = self._make_query_broader(query)

                raw_results = self.kb_manager.retrieve(
                    kb_id=self.kb_id,
                    query=search_query,
                    max_results=config.RETRIEVAL_K
                )

                results: List[RetrievalResult] = []
                for item in raw_results:
                    content_text = item.get("content", {}).get("text", "") or ""
                    score = item.get("score", 0.0) or 0.0
                    metadata = item.get("metadata", {}) or {}
                    source_uri = (
                        item.get("location", {})
                            .get("s3Location", {})
                            .get("uri", "")
                        or ""
                    )

                    # chunk_id may not exist in KB output - fallback to hashed key
                    chunk_id = metadata.get("chunk_id")
                    if not chunk_id:
                        chunk_id = hashlib.md5((source_uri + content_text[:200]).encode()).hexdigest()[:16]

                    results.append(RetrievalResult(
                        content=content_text,
                        score=score,
                        metadata=metadata,
                        source=source_uri,
                        chunk_id=chunk_id
                    ))

                all_results.extend(results)

                # Break early if strong hit
                if results and max(r.score for r in results) > 0.7:
                    break

            except Exception as e:
                self.logger.warning(f"Retrieval attempt {attempt + 1} failed: {e}")
                continue

        # Deduplicate results by (source + chunk_id)
        unique: Dict[str, RetrievalResult] = {}
        for r in all_results:
            key = f"{r.source}::{r.chunk_id}"
            if key not in unique or r.score > unique[key].score:
                unique[key] = r

        sorted_results = sorted(unique.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:config.RETRIEVAL_K]

    def _make_query_specific(self, query: str) -> str:
        return f"{query} specific detailed exact"

    def _make_query_broader(self, query: str) -> str:
        broad_query = re.sub(r"\b(specific|exactly|precisely)\b", "", query, flags=re.IGNORECASE)
        return f"{broad_query.strip()} overview general information"

    # -------------------------
    # Context building
    # -------------------------

    def _prepare_enhanced_context(self, query: str, retrieval_results: List[RetrievalResult]) -> str:
        context = "# Context\n\n"

        # Conversation history
        memory_messages = self.memory.chat_memory.messages
        if memory_messages:
            context += "## Previous conversation\n"
            max_history_tokens = 800
            used_tokens = 0

            for msg in reversed(memory_messages[-6:]):
                role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                msg_content = msg.content
                msg_tokens = len(msg_content) // 4

                if used_tokens + msg_tokens > max_history_tokens:
                    allowed_chars = max(0, (max_history_tokens - used_tokens) * 4)
                    msg_content = msg_content[:allowed_chars] + "..."

                context += f"- **{role}:** {msg_content}\n"
                used_tokens += msg_tokens

                if used_tokens >= max_history_tokens:
                    break

            context += "\n"

        # Retrieved chunks
        context += "## Retrieved knowledge\n\n"
        remaining_tokens = self.context_window_tokens - (len(context) // 4) - 600

        for i, r in enumerate(retrieval_results):
            if remaining_tokens <= 0:
                break

            header = f"### Chunk {i+1} (score={r.score:.3f})\n"
            meta_parts = []
            if r.metadata:
                for k in ["file_name", "page", "sheet", "slide"]:
                    if k in r.metadata:
                        meta_parts.append(f"{k}={r.metadata[k]}")
            meta_line = f"Metadata: {', '.join(meta_parts)}\n" if meta_parts else ""
            src_line = f"Source: {r.source}\n" if r.source else ""

            block = header + meta_line + src_line

            content_budget = remaining_tokens - (len(block) // 4)
            chunk_text = r.content

            if content_budget <= 0:
                break

            if (len(chunk_text) // 4) > content_budget:
                chunk_text = chunk_text[:content_budget * 4] + "..."

            block += f"Content:\n{chunk_text}\n\n"

            context += block
            remaining_tokens -= len(block) // 4

        # Instructions
        context += f"## Question\n{query}\n\n"
        context += (
            "## Instructions\n"
            "- Answer using ONLY the retrieved knowledge if possible.\n"
            "- Cite sources clearly (file/page/sheet/slide).\n"
            "- If the documents do not contain the answer, say so.\n"
            "- Be precise and factual.\n"
        )

        return context

    # -------------------------
    # Streaming generation (Claude)
    # -------------------------

    def _stream_generate_enhanced(
        self,
        context: str,
        query: str,
        retrieval_results: List[RetrievalResult],
    ) -> Iterator[Dict[str, Any]]:

        try:
            # Messages API prompt format (NO Human:/Assistant: prefixes)
            prompt = f"""{context}

Please answer the question comprehensively, citing sources when possible.

Question: {query}
"""

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 250,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            }

            response = self.bedrock_runtime.invoke_model_with_response_stream(
                modelId=config.BEDROCK_MODEL_ID,
                body=json.dumps(body),
            )

            full_text = ""
            last_conf_emit_words = 0
            CONF_EMIT_EVERY_N_WORDS = 40

            for event in response.get("body", []):
                if "chunk" not in event or "bytes" not in event["chunk"]:
                    continue

                try:
                    chunk = json.loads(event["chunk"]["bytes"])
                except Exception:
                    continue

                chunk_type = chunk.get("type")

                if chunk_type == "content_block_delta":
                    delta = chunk.get("delta", {})
                    text_chunk = delta.get("text", "")
                    if not text_chunk:
                        continue

                    full_text += text_chunk
                    yield {"type": "content", "content": text_chunk}

                    word_count = len(full_text.split())
                    sentence_count = len(re.findall(r"[.!?]+", full_text))

                    if (word_count - last_conf_emit_words) >= CONF_EMIT_EVERY_N_WORDS:
                        conf = self._calculate_enhanced_confidence(
                            full_text, word_count, sentence_count, retrieval_results
                        )
                        yield {"type": "confidence", "score": conf}
                        last_conf_emit_words = word_count

                elif chunk_type == "message_stop":
                    word_count = len(full_text.split())
                    sentence_count = len(re.findall(r"[.!?]+", full_text))

                    conf = self._calculate_enhanced_confidence(
                        full_text, word_count, sentence_count, retrieval_results
                    )
                    yield {"type": "confidence", "score": conf}

                    sources = self._extract_enhanced_sources(full_text, retrieval_results)
                    yield {"type": "sources", "sources": sources}
                    break

                elif chunk_type == "error":
                    msg = chunk.get("message", "Unknown model stream error")
                    yield {"type": "error", "content": msg}
                    break

        except Exception as e:
            self.logger.error(f"Enhanced streaming error: {e}", exc_info=True)
            yield {"type": "error", "content": f"Streaming error: {str(e)}"}

    # -------------------------
    # Metrics / scoring helpers
    # -------------------------

    def _calculate_enhanced_confidence(
        self,
        text: str,
        word_count: int,
        sentence_count: int,
        retrieval_results: List[RetrievalResult]
    ) -> float:

        confidence = 0.5

        if word_count > 50:
            confidence += 0.1
        if word_count > 100:
            confidence += 0.1
        if word_count > 200:
            confidence += 0.1

        if sentence_count > 0:
            avg_words = word_count / max(sentence_count, 1)
            if 10 <= avg_words <= 25:
                confidence += 0.1

        text_lower = text.lower()

        uncertainty_phrases = [
            "i'm not sure", "i don't know", "unclear", "uncertain",
            "might be", "could be", "possibly", "perhaps", "maybe",
            "i think", "i believe", "seems like"
        ]
        uncertainty_count = sum(1 for p in uncertainty_phrases if p in text_lower)
        confidence -= uncertainty_count * 0.1

        confident_phrases = [
            "according to", "based on", "the document states", "as shown in",
            "clearly", "specifically", "exactly", "definitely", "certainly",
            "the data shows", "research indicates", "studies show"
        ]
        confident_count = sum(1 for p in confident_phrases if p in text_lower)
        confidence += confident_count * 0.08

        citation_indicators = ["source:", "file:", "page:", "sheet:", "slide:"]
        citation_count = sum(1 for ind in citation_indicators if ind in text_lower)
        confidence += min(citation_count * 0.05, 0.15)

        if retrieval_results:
            avg_retrieval_score = sum(r.score for r in retrieval_results) / len(retrieval_results)
            confidence += avg_retrieval_score * 0.2

        factual_patterns = [r"\d+%", r"\$\d+", r"\b\d{4}\b"]
        factual_count = sum(1 for p in factual_patterns if re.search(p, text_lower))
        confidence += min(factual_count * 0.03, 0.1)

        return max(0.0, min(1.0, confidence))

    def _extract_enhanced_sources(
        self,
        response_text: str,
        retrieval_results: List[RetrievalResult]
    ) -> List[str]:

        sources: List[str] = []

        # If model explicitly mentions sources, keep them
        for line in response_text.splitlines():
            line_lower = line.lower()
            if any(k in line_lower for k in ["source:", "file:", "page:", "sheet:", "slide:", "according to"]):
                sources.append(line.strip())

        # Also add top retrieval sources
        for r in retrieval_results[:3]:
            meta = r.metadata or {}
            info = []
            if "file_name" in meta:
                info.append(f"File: {meta['file_name']}")
            if "page" in meta:
                info.append(f"Page: {meta['page']}")
            if "sheet" in meta:
                info.append(f"Sheet: {meta['sheet']}")
            if "slide" in meta:
                info.append(f"Slide: {meta['slide']}")
            if not info and r.source:
                info.append(f"Source: {r.source}")

            if info:
                sources.append(" | ".join(info))

        # Dedup preserve order
        seen = set()
        unique_sources = []
        for s in sources:
            if s and s not in seen and len(s.strip()) > 5:
                unique_sources.append(s)
                seen.add(s)

        return unique_sources[:5]

    def _calculate_retrieval_score(self, retrieval_results: List[Dict]) -> float:
        if not retrieval_results:
            return 0.0

        scores = [r.get("score", 0.0) for r in retrieval_results]
        weights = [1.0, 0.8, 0.6, 0.4, 0.2][:len(scores)]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return weighted_sum / sum(weights)

    def _calculate_context_relevance(self, response: str, retrieval_results: List[RetrievalResult]) -> float:
        if not retrieval_results or not response:
            return 0.0

        response_words = set(response.lower().split())
        total = 0.0

        for r in retrieval_results:
            content_words = set(r.content.lower().split())
            if not content_words:
                continue

            overlap = len(content_words.intersection(response_words))
            union = len(content_words.union(response_words))
            total += overlap / union if union else 0.0

        return min(total / len(retrieval_results), 1.0)

    # -------------------------
    # Memory / analytics helpers
    # -------------------------

    def _update_memory(self, query: str, response: str) -> None:
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response)

    def _track_query_patterns(self, query: str) -> None:
        words = re.findall(r"\b\w+\b", query.lower())
        key_terms = [w for w in words if len(w) > 3]
        for t in key_terms[:5]:
            self.query_patterns[t] += 1

    def _generate_cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()[:16]

    def _cache_response(self, cache_key: str, content: str, sources: List[str], metrics: Dict[str, Any]) -> None:
        self.response_cache[cache_key] = {
            "content": content,
            "sources": sources,
            "metrics": metrics,
            "timestamp": datetime.now(),
            "access_count": 0
        }

        # Cache size limit
        if len(self.response_cache) > 100:
            oldest_key = min(self.response_cache, key=lambda k: self.response_cache[k]["timestamp"])
            del self.response_cache[oldest_key]

    def _update_enhanced_metrics(
        self,
        confidence_score: float = 0.0,
        retrieval_score: float = 0.0,
        response_time: float = 0.0,
        sources_count: int = 0,
        token_count: int = 0,
        context_relevance: float = 0.0,
        retrieval_success: bool = True,
    ) -> None:

        self.accuracy_metrics.total_queries += 1
        n = self.accuracy_metrics.total_queries

        def running_avg(old, new):
            return (old * (n - 1) + new) / n

        if confidence_score > 0:
            self.accuracy_metrics.avg_confidence = running_avg(self.accuracy_metrics.avg_confidence, confidence_score)

        if retrieval_score > 0:
            self.accuracy_metrics.avg_retrieval_score = running_avg(self.accuracy_metrics.avg_retrieval_score, retrieval_score)

        if response_time > 0:
            self.accuracy_metrics.avg_response_time = running_avg(self.accuracy_metrics.avg_response_time, response_time)

        if token_count > 0:
            self.accuracy_metrics.avg_token_count = running_avg(self.accuracy_metrics.avg_token_count, token_count)

        if context_relevance > 0:
            self.accuracy_metrics.context_relevance_score = running_avg(
                self.accuracy_metrics.context_relevance_score, context_relevance
            )

        if sources_count > 0:
            current_coverage = min(1.0, sources_count / config.RETRIEVAL_K)
            self.accuracy_metrics.source_coverage = running_avg(self.accuracy_metrics.source_coverage, current_coverage)

        if retrieval_success:
            self.accuracy_metrics.successful_retrievals += 1
        else:
            self.accuracy_metrics.failed_retrievals += 1

        self.performance_history.append({
            "timestamp": datetime.now(),
            "confidence": confidence_score,
            "retrieval_score": retrieval_score,
            "response_time": response_time,
            "context_relevance": context_relevance
        })

        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    # -------------------------
    # User feedback + reporting
    # -------------------------

    def add_user_feedback(self, positive: bool, feedback_text: Optional[str] = None) -> None:
        if positive:
            self.accuracy_metrics.user_feedback_positive += 1
        else:
            self.accuracy_metrics.user_feedback_negative += 1

        self.logger.info(f"User feedback: {'positive' if positive else 'negative'}")
        if feedback_text:
            self.logger.info(f"Feedback text: {feedback_text}")

    def get_accuracy_report(self) -> Dict[str, Any]:
        total_feedback = self.accuracy_metrics.user_feedback_positive + self.accuracy_metrics.user_feedback_negative

        user_satisfaction = (
            self.accuracy_metrics.user_feedback_positive / total_feedback
            if total_feedback else 0.0
        )

        total_retrievals = self.accuracy_metrics.successful_retrievals + self.accuracy_metrics.failed_retrievals

        retrieval_success_rate = (
            self.accuracy_metrics.successful_retrievals / total_retrievals
            if total_retrievals else 0.0
        )

        recent = self.performance_history[-10:] if len(self.performance_history) >= 10 else []
        confidence_trend = "stable"
        if len(recent) >= 5:
            recent_conf = [p["confidence"] for p in recent[-5:]]
            earlier_conf = [p["confidence"] for p in recent[:5]]
            if sum(recent_conf) > sum(earlier_conf) * 1.1:
                confidence_trend = "improving"
            elif sum(recent_conf) < sum(earlier_conf) * 0.9:
                confidence_trend = "declining"

        return {
            "total_queries": self.accuracy_metrics.total_queries,
            "average_confidence": round(self.accuracy_metrics.avg_confidence, 3),
            "average_retrieval_score": round(self.accuracy_metrics.avg_retrieval_score, 3),
            "average_response_time": round(self.accuracy_metrics.avg_response_time, 2),
            "average_token_count": round(self.accuracy_metrics.avg_token_count, 1),
            "source_coverage": round(self.accuracy_metrics.source_coverage, 3),
            "context_relevance_score": round(self.accuracy_metrics.context_relevance_score, 3),
            "user_satisfaction": round(user_satisfaction, 3),
            "retrieval_success_rate": round(retrieval_success_rate, 3),
            "total_feedback": total_feedback,
            "positive_feedback": self.accuracy_metrics.user_feedback_positive,
            "negative_feedback": self.accuracy_metrics.user_feedback_negative,
            "session_id": self.session_id,
            "confidence_trend": confidence_trend,
            "cache_size": len(self.response_cache),
            "top_query_terms": dict(sorted(self.query_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
        }

    def export_chat_history(self) -> List[Dict]:
        return [asdict(msg) for msg in self.chat_history]

    def export_analytics(self) -> Dict[str, Any]:
        return {
            "accuracy_metrics": asdict(self.accuracy_metrics),
            "performance_history": self.performance_history,
            "query_patterns": dict(self.query_patterns),
            "session_info": {
                "session_id": self.session_id,
                "memory_window": self.memory.k,
                "cache_enabled": self.enable_caching,
                "query_expansion_enabled": self.enable_query_expansion
            }
        }

    def clear_memory(self) -> None:
        self.memory.clear()
        self.chat_history.clear()
        self.session_id = str(uuid.uuid4())
        self.response_cache.clear()
        self.logger.info("Memory, chat history, and cache cleared")

    def optimize_performance(self) -> None:
        # Adjust memory window based on usage
        if self.accuracy_metrics.total_queries > 50:
            avg_conf = self.accuracy_metrics.avg_confidence
            if avg_conf < 0.6:
                self.memory.k = min(self.memory.k + 2, 15)
                self.logger.info(f"Increased memory window to {self.memory.k}")
            elif avg_conf > 0.8:
                self.memory.k = max(self.memory.k - 1, 5)
                self.logger.info(f"Decreased memory window to {self.memory.k}")

        # Clean old cache entries
        cutoff = datetime.now() - timedelta(hours=24)
        old_keys = [k for k, v in self.response_cache.items() if v["timestamp"] < cutoff]
        for k in old_keys:
            del self.response_cache[k]

        if old_keys:
            self.logger.info(f"Cleaned {len(old_keys)} old cache entries")
