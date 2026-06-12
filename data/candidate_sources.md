# Candidate Dataset Sources

Downloaded on 2026-06-07 for dataset feasibility research.

The actual files live under `dataset/candidate/`, which is ignored by Git.

## LoCoMO

Official repository: <https://github.com/snap-research/locomo>

| Local file | Source URL | Size | SHA-256 |
| --- | --- | ---: | --- |
| `dataset/candidate/locomo/locomo10.json` | `https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json` | 2,805,274 bytes | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |

## LongMemEval Cleaned

Official repository: <https://github.com/xiaowu0162/LongMemEval>

HuggingFace dataset: <https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned>

| Local file | Source URL | Size | SHA-256 |
| --- | --- | ---: | --- |
| `dataset/candidate/longmemeval/longmemeval_oracle.json` | `https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json` | 15,388,478 bytes | `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c` |
| `dataset/candidate/longmemeval/longmemeval_s_cleaned.json` | `https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json` | 277,383,467 bytes | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `dataset/candidate/longmemeval/longmemeval_m_cleaned.json` | `https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json` | 2,737,100,077 bytes | `9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f` |

## MemoryAgentBench

Official repository: <https://github.com/HUST-AI-HYZ/MemoryAgentBench>

HuggingFace dataset: <https://huggingface.co/datasets/ai-hyz/MemoryAgentBench>

| Local file | Source URL | Size | SHA-256 |
| --- | --- | ---: | --- |
| `dataset/candidate/MemoryAgentBench/Accurate_Retrieval.parquet` | `https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/Accurate_Retrieval.parquet` | 20,024,386 bytes | `56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1` |
| `dataset/candidate/MemoryAgentBench/Conflict_Resolution.parquet` | `https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/Conflict_Resolution.parquet` | 1,491,588 bytes | `24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45` |
| `dataset/candidate/MemoryAgentBench/Long_Range_Understanding.parquet` | `https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/Long_Range_Understanding.parquet` | 49,342,452 bytes | `5ab175461954db67770d4a4cb69e569b513ebb96aceb9ee79b57f67488bcd539` |
| `dataset/candidate/MemoryAgentBench/Test_Time_Learning.parquet` | `https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/Test_Time_Learning.parquet` | 3,947,476 bytes | `5338753be48f925d03318eed66117286e3489025fabe050a547bd086cd7d79c0` |

### EventQA Workload Note

EventQA appears inside `Accurate_Retrieval.parquet` as `eventqa_full`,
`eventqa_65536`, and `eventqa_131072`. It is worth keeping for VisPage-style
evaluation because each row naturally forms a fixed-memory, sequential-query
scenario:

1. The long `context` is the fixed memory corpus, similar to a long novel or
   case file already loaded into memory.
2. The row contains 100 ordered questions.
3. Question `i` includes the first `i + 1` known events and asks the model to
   choose the next event from several candidates.
4. The answer to question `i` appears as an additional known event in question
   `i + 1`.

This is analogous to a user repeatedly asking "what happens next?" while moving
through the same long record. The dataset itself does not require runtime memory
insertion: the corpus remains fixed, and the next query already contains the
expanded event prefix. For evaluation, treat each EventQA row as one scenario:
construct pages from `context`, then replay `questions` in file order.

The expected value for VisPage is query locality. Adjacent questions are about
the same narrative chain, so useful pages are likely to overlap or remain nearby.
This makes EventQA more suitable for page/cache reuse testing than randomly
ordered independent QA. The caveat is that its locality is strong and
template-like, so report it as a "stateful narrative query stream over fixed
memory" rather than as a generic database workload.

## Mem-PAL / PAL-Set

Official repository: <https://github.com/hzp3517/Mem-PAL>

Paper: <https://arxiv.org/abs/2511.13410>

Local clone commit: `ad290d4a1d39932531e74fbc137be865aceea9ba`

| Local file | Source path | Size | SHA-256 |
| --- | --- | ---: | --- |
| `dataset/candidate/Mem-PAL/data_synthesis_v2/data/input.json` | `data_synthesis_v2/data/input.json` | 73,431,492 bytes | `8a20ff185acbac1dd91a852af1e235f958ed44159ba4b07394cbda07bcf56a9b` |
| `dataset/candidate/Mem-PAL/data_synthesis_v2/data_en/input_en.json` | `data_synthesis_v2/data_en/input_en.json` | 82,566,253 bytes | `86784402bb3a18589250c32fd47f7ea1100bc04882e7dd7fdec6eeb6eb0e311a` |
| `dataset/candidate/Mem-PAL/data_synthesis_v2/data/background/background.json` | `data_synthesis_v2/data/background/background.json` | 102,446 bytes | `a28d8d62a0a92942b6d57a29b74cdd444022139e6803a920f352fff3d99ab378` |
| `dataset/candidate/Mem-PAL/data_synthesis_v2/data/situation/situation.json` | `data_synthesis_v2/data/situation/situation.json` | 4,290,120 bytes | `c33db48221be32917c5acf68957a251cc115d02f54770c8afe2364cbe511648a` |
| `dataset/candidate/Mem-PAL/data_synthesis_v2/data/preference/preference.json` | `data_synthesis_v2/data/preference/preference.json` | 1,806,012 bytes | `50549f00699e87eb0e98f108e857f65c7bfe843d6bc573352fa61cab94aada1d` |
| `dataset/candidate/Mem-PAL/data_synthesis_v2/data/requirement/requirement.json` | `data_synthesis_v2/data/requirement/requirement.json` | 111,135 bytes | `7d7055300d878c1d4000d53744fb0e1a05305db1bcdf853aee3cb0d86af3fbda` |

PAL-Set is a Chinese synthetic long-term user-agent interaction dataset with
100 users. Each user has a chronological `history` split and a chronological
`query` split. `history` contains earlier sessions and should be used as the
fixed memory corpus. `query` contains the final-month sessions and can be
replayed in timestamp order as a user-specific query workload.

Initial local stats for the Chinese original:

| Split | Sessions | Logs | Dialogue turns | Topics |
| --- | ---: | ---: | ---: | ---: |
| `history` | 2,564 | 88,872 | 36,166 | 6,252 |
| `query` | 326 | 10,747 | 3,927 | 826 |

The single-turn PAL-Bench tasks use each query-session topic as an evaluation
sample: `user_query` for requirement restatement, `requirement` plus candidate
solutions for solution selection/proposal. This makes PAL-Set a promising
candidate for fixed-memory personalized query workloads. The main adaptation
choice is whether query-session logs are treated as part of the current query
context, added to memory incrementally before each session, or excluded to keep
the memory corpus strictly fixed.

## Conversation Chronicles

HuggingFace dataset: <https://huggingface.co/datasets/jihyoung/ConversationChronicles>

Paper: <https://aclanthology.org/2023.emnlp-main.838>

| Local file | Source URL | Size | SHA-256 |
| --- | --- | ---: | --- |
| `dataset/candidate/ConversationChronicles/test.jsonl` | `https://huggingface.co/datasets/jihyoung/ConversationChronicles/resolve/main/test.jsonl` | 168,658,644 bytes | `eef3ea2961cb4ba3e53849b5a58349b9fa64a106530ad41427e29c1d78c01cfe` |

The official dataset has 200,000 episodes and 1,000,000 sessions. Each episode
contains exactly five chronological sessions between two speakers, a
relationship label, five time-interval labels, and session summaries. The
downloaded `test.jsonl` split has 20,000 episodes and 100,000 sessions.

Initial local stats for `test.jsonl`:

| Item | Value |
| --- | ---: |
| Episodes | 20,000 |
| Sessions | 100,000 |
| Avg turns/session | 11.94 |
| Avg chars/session | 1,077 |
| Avg pair-query opportunities/episode, session 5 only | 11.95 |
| Avg pair-query opportunities/episode, sessions 2-5 | 43.79 |

Conversation Chronicles is not a QA dataset. It is best viewed as a
multi-session response-generation workload. A natural VisPage adaptation is to
use sessions 1-4 as fixed memory and replay user turns in session 5 as ordered
queries with the next speaker response as the target. A longer trace variant can
replay sessions 2-5 while revealing only prior sessions and prior turns within
the current session.

This dataset is promising for profiling whether temporal locality and semantic
locality align: every episode has explicit time gaps and a speaker relationship,
and later sessions frequently refer back to earlier sessions. The main caveat is
that each episode is short: only five sessions and roughly 12 turns per session.
It therefore offers many short traces rather than a small number of long,
single-user traces.

## Local Validation

All downloaded JSON files passed `jq -e .`.
All MemoryAgentBench parquet files were readable with `pyarrow`.
Mem-PAL/PAL-Set JSON files were parsed successfully with Python `json`.
Conversation Chronicles `test.jsonl` was parsed successfully with Python
`json`.
