import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, cast

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore[assignment]


@dataclass
class BoundaryPoint:
    position: int
    similarity: float
    dissimilarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仅执行 LoCoMo 对话轮次分割，并保存可视化结果 JSON"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="LoCoMo 数据集 JSON 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/locomo_turn_segments.json",
        help="分割结果输出 JSON 路径",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="用于 turn 语义相似度的 embedding 模型",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "sentence-transformers", "tfidf"],
        default="auto",
        help="相似度计算后端：auto 优先 sentence-transformers，不可用时回退 tfidf",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="最多处理多少条样本（默认 10，便于可视化调试）",
    )
    parser.add_argument(
        "--quantile-threshold",
        type=float,
        default=0.75,
        help="候选边界分位阈值，基于 dissimilarity 分数",
    )
    parser.add_argument(
        "--absolute-threshold",
        type=float,
        default=0.42,
        help="候选边界绝对阈值，基于 dissimilarity 分数",
    )
    parser.add_argument(
        "--min-segment-turns",
        type=int,
        default=3,
        help="最小分段长度（turn 数）",
    )
    return parser.parse_args()


def parse_locomo_timestamp(timestamp_str: str) -> str:
    ts = timestamp_str.strip("()")
    try:
        dt = datetime.strptime(ts, "%I:%M %p on %d %B, %Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp_str


def extract_sessions(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    speaker_a = conversation.get("speaker_a", "Speaker_A")
    speaker_b = conversation.get("speaker_b", "Speaker_B")

    session_ids: List[int] = []
    for key in conversation.keys():
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                session_ids.append(int(key.split("_")[1]))
            except Exception:
                continue

    sessions: List[Dict[str, Any]] = []
    for sid in sorted(set(session_ids)):
        session_key = f"session_{sid}"
        timestamp_key = f"{session_key}_date_time"

        raw_turns = conversation.get(session_key, [])
        if not raw_turns:
            continue

        turns: List[Dict[str, Any]] = []
        for i, turn in enumerate(raw_turns):
            speaker = turn.get("speaker", "Unknown")
            text = str(turn.get("text", "")).strip()
            blip_caption = turn.get("blip_caption")
            if blip_caption:
                text = f"{text} (image description: {blip_caption})"

            if speaker == speaker_a:
                speaker_tag = "speaker_a"
            elif speaker == speaker_b:
                speaker_tag = "speaker_b"
            else:
                speaker_tag = "unknown"

            turns.append(
                {
                    "turn_index": i,
                    "speaker": speaker,
                    "speaker_tag": speaker_tag,
                    "text": text,
                    "dia_id": turn.get("dia_id"),
                }
            )

        sessions.append(
            {
                "session_id": session_key,
                "timestamp": parse_locomo_timestamp(str(conversation.get(timestamp_key, ""))),
                "turns": turns,
            }
        )

    return sessions


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized = embeddings / norms
    return normalized @ normalized.T


class EmbeddingBackend(Protocol):
    def encode(self, sentences: List[str], normalize_embeddings: bool = False) -> Any:
        ...


def tfidf_adjacent_scores(texts: List[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
    )
    mat = vectorizer.fit_transform(texts)

    sims: List[float] = []
    for i in range(len(texts) - 1):
        v1 = mat.getrow(i)
        v2 = mat.getrow(i + 1)
        numerator = float(v1.dot(v2.T).toarray()[0][0])
        denom = float(np.linalg.norm(v1.toarray()) * np.linalg.norm(v2.toarray()))
        if denom <= 1e-12:
            sims.append(0.0)
        else:
            sims.append(numerator / denom)
    return np.array(sims, dtype=np.float32)


def compute_adjacent_similarities(
    texts: List[str],
    backend: str,
    embedder: Optional[Any],
) -> np.ndarray:
    if len(texts) <= 1:
        return np.array([], dtype=np.float32)

    use_tfidf = backend == "tfidf"
    if backend == "sentence-transformers" and embedder is None:
        raise RuntimeError(
            "backend=sentence-transformers 但当前环境缺少 sentence_transformers，请先安装或使用 --backend tfidf"
        )
    if backend == "auto" and embedder is None:
        use_tfidf = True

    if use_tfidf:
        return tfidf_adjacent_scores(texts)

    if embedder is None:
        raise RuntimeError("当前后端需要 embedding 模型，但 embedder 为空")

    typed_embedder = cast(EmbeddingBackend, embedder)
    embs = typed_embedder.encode(texts, normalize_embeddings=False)
    embs_np = np.asarray(embs, dtype=np.float32)
    sim_matrix = cosine_similarity_matrix(embs_np)
    return np.array([sim_matrix[i, i + 1] for i in range(len(texts) - 1)], dtype=np.float32)


def pick_boundaries(
    dissimilarities: np.ndarray,
    similarities: np.ndarray,
    quantile_threshold: float,
    absolute_threshold: float,
    min_segment_turns: int,
) -> List[BoundaryPoint]:
    if len(dissimilarities) == 0:
        return []

    quantile_threshold = float(np.clip(quantile_threshold, 0.0, 1.0))
    dyn_threshold = float(np.quantile(dissimilarities, quantile_threshold))
    threshold = max(dyn_threshold, absolute_threshold)

    candidates: List[int] = []
    for idx, score in enumerate(dissimilarities):
        left = dissimilarities[idx - 1] if idx > 0 else -1.0
        right = dissimilarities[idx + 1] if idx < len(dissimilarities) - 1 else -1.0
        is_local_peak = score >= left and score >= right
        if is_local_peak and score >= threshold:
            candidates.append(idx + 1)

    # 最小段长约束：边界之间至少间隔 min_segment_turns。
    kept: List[BoundaryPoint] = []
    last_boundary = 0
    n_turns = len(dissimilarities) + 1

    for b in candidates:
        if b - last_boundary < min_segment_turns:
            continue
        if n_turns - b < min_segment_turns:
            continue
        kept.append(
            BoundaryPoint(
                position=b,
                similarity=float(similarities[b - 1]),
                dissimilarity=float(dissimilarities[b - 1]),
            )
        )
        last_boundary = b

    return kept


def build_segments(turns: List[Dict[str, Any]], boundaries: List[BoundaryPoint]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    starts = [0] + [b.position for b in boundaries]
    ends = [b.position for b in boundaries] + [len(turns)]

    for idx, (start, end) in enumerate(zip(starts, ends)):
        slice_turns = turns[start:end]
        speaker_counter = Counter(t["speaker"] for t in slice_turns)
        preview_text = " ".join(t["text"] for t in slice_turns[:2]).strip()
        if len(preview_text) > 160:
            preview_text = preview_text[:157] + "..."

        segments.append(
            {
                "segment_id": idx,
                "turn_start": start,
                "turn_end": end,
                "turn_count": max(end - start, 0),
                "speaker_distribution": dict(speaker_counter),
                "preview": preview_text,
            }
        )

    return segments


def segment_one_session(
    session: Dict[str, Any],
    embedder: Optional[Any],
    backend: str,
    quantile_threshold: float,
    absolute_threshold: float,
    min_segment_turns: int,
) -> Dict[str, Any]:
    turns = session["turns"]
    texts = [t["text"] if t["text"] else "[EMPTY]" for t in turns]

    if len(texts) <= 1:
        boundaries: List[BoundaryPoint] = []
        adjacent_scores: List[Dict[str, Any]] = []
    else:
        similarities = compute_adjacent_similarities(texts=texts, backend=backend, embedder=embedder)
        dissimilarities = 1.0 - similarities

        boundaries = pick_boundaries(
            dissimilarities=dissimilarities,
            similarities=similarities,
            quantile_threshold=quantile_threshold,
            absolute_threshold=absolute_threshold,
            min_segment_turns=min_segment_turns,
        )

        adjacent_scores = [
            {
                "left_turn": i,
                "right_turn": i + 1,
                "similarity": float(similarities[i]),
                "dissimilarity": float(dissimilarities[i]),
            }
            for i in range(len(similarities))
        ]

    segments = build_segments(turns, boundaries)

    return {
        "session_id": session["session_id"],
        "timestamp": session["timestamp"],
        "num_turns": len(turns),
        "turns": turns,
        "boundaries": [
            {
                "position": b.position,
                "after_turn": b.position - 1,
                "before_turn": b.position,
                "similarity": b.similarity,
                "dissimilarity": b.dissimilarity,
            }
            for b in boundaries
        ],
        "adjacent_scores": adjacent_scores,
        "segments": segments,
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("LoCoMo 输入文件应为 JSON 列表")

    max_samples = max(args.max_samples, 1)
    selected_samples = raw_data[:max_samples]

    embedder: Optional[Any] = None
    used_backend = args.backend

    if args.backend in {"auto", "sentence-transformers"}:
        if SentenceTransformer is None:
            if args.backend == "sentence-transformers":
                raise RuntimeError("未检测到 sentence_transformers，请安装后再运行，或使用 --backend tfidf")
            used_backend = "tfidf"
            print("[Info] sentence_transformers 不可用，自动回退到 TF-IDF 后端。")
        else:
            embedder = SentenceTransformer(args.embedding_model)
            used_backend = "sentence-transformers"

    output_samples: List[Dict[str, Any]] = []
    for idx, item in enumerate(selected_samples):
        conversation = item.get("conversation", {})
        sample_id = item.get("sample_id", f"sample_{idx}")
        speaker_a = conversation.get("speaker_a", "Speaker_A")
        speaker_b = conversation.get("speaker_b", "Speaker_B")

        sessions = extract_sessions(conversation)
        segmented_sessions = [
            segment_one_session(
                session=s,
                embedder=embedder,
                backend=used_backend,
                quantile_threshold=args.quantile_threshold,
                absolute_threshold=args.absolute_threshold,
                min_segment_turns=args.min_segment_turns,
            )
            for s in sessions
        ]

        output_samples.append(
            {
                "sample_id": sample_id,
                "speakers": {
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                },
                "num_sessions": len(segmented_sessions),
                "sessions": segmented_sessions,
            }
        )

    result = {
        "meta": {
            "dataset": "locomo",
            "input_path": str(input_path),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_model": args.embedding_model,
            "similarity_backend": used_backend,
            "max_samples": max_samples,
            "quantile_threshold": args.quantile_threshold,
            "absolute_threshold": args.absolute_threshold,
            "min_segment_turns": args.min_segment_turns,
            "num_samples_processed": len(output_samples),
        },
        "samples": output_samples,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"分割完成，结果已保存: {output_path}")
    print(f"处理样本数: {len(output_samples)}")


if __name__ == "__main__":
    main()
