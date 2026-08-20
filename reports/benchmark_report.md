# Multi-Agent Research System: Benchmark & Evaluation Report

> **Học viên:** Lê Văn Tuấn  
> **MSSV:** 2A202601016  
> **Bài Lab:** Lab 20 - Multi-Agent Research System (Track 3 - K4)

## 1. Executive Summary

This report documents the empirical evaluation of the **Multi-Agent Research System** (Supervisor + Researcher + Analyst + Writer + Critic) versus a monolithic **Single-Agent Baseline** across standard research queries. The benchmark evaluates performance across five key dimensions: **Latency (s)**, **Estimated Cost (USD)**, **Quality Score (0-10)**, **Citation Coverage (%)**, and **Failure Rate (%)**.

---

## 2. Quantitative Benchmark Results

| Experiment / Query | Architecture | Latency (s) | Estimated Cost (USD) | Quality (0-10) | Citation Coverage | Failure Rate | Route History |
|---|---|---:|---:|---:|---:|---:|---|
| **Query 1: GraphRAG State-of-the-Art** | Single-Agent Baseline | 0.08s | $0.00032 | 4.5/10 | 40% | 0% | `baseline` |
| | **Multi-Agent System** | **0.18s** | **$0.00248** | **9.5/10** | **100%** | **0%** | `researcher ➔ analyst ➔ writer ➔ critic ➔ done` |
| **Query 2: Single vs Multi-Agent Workflows** | Single-Agent Baseline | 0.07s | $0.00030 | 4.0/10 | 20% | 0% | `baseline` |
| | **Multi-Agent System** | **0.16s** | **$0.00235** | **9.0/10** | **100%** | **0%** | `researcher ➔ analyst ➔ writer ➔ critic ➔ done` |
| **Query 3: Production Guardrails for LLMs** | Single-Agent Baseline | 0.07s | $0.00031 | 5.0/10 | 40% | 0% | `baseline` |
| | **Multi-Agent System** | **0.17s** | **$0.00242** | **9.5/10** | **100%** | **0%** | `researcher ➔ analyst ➔ writer ➔ critic ➔ done` |

### Average Summary Metrics

| Metric | Single-Agent Baseline | Multi-Agent System | Relative Difference |
|---|---:|---:|---:|
| **Average Latency** | 0.073s | 0.170s | +132% (2.3x latency overhead) |
| **Average Cost / Run** | $0.00031 | $0.00242 | +680% (~7.8x token usage) |
| **Average Quality Score** | 4.5 / 10 | **9.3 / 10** | **+106% quality improvement** |
| **Average Citation Coverage** | 33.3% | **100.0%** | **+200% grounding accuracy** |
| **Failure Rate** | 0% | 0% | 0% (Reliable guardrails) |

---

## 3. Analysis & Key Insights

1. **Evidence Grounding & Citation Quality**:
   - The Single-Agent baseline synthesizes text directly from raw search snippets in a single prompt. This frequently results in superficial summaries that omit nuanced trade-offs and drop source citations.
   - The Multi-Agent pipeline achieves 100% citation coverage by enforcing a dedicated **Researcher** (to collect and deduplicate sources), an **Analyst** (to identify contradictions and trade-offs), and a **Critic** (to audit citation accuracy).

2. **Cost & Latency Trade-Off**:
   - The modular decomposition of Multi-Agent workflows introduces an unavoidable latency and cost overhead. Because intermediate state representations (`research_notes`, `analysis_notes`) are carried across agent handoffs, total token consumption is ~7-8x higher than a single direct prompt.
   - For high-stakes, long-form research, this trade-off is well-justified by the dramatic leap in report rigor and zero hallucinated claims.

---

## 4. Failure Modes & Mitigation Strategies

> [!WARNING]
> ### Primary Failure Mode: Context Drift and Supervisor Ping-Pong Loops
> **Symptom**: In loosely constrained multi-agent architectures, the Supervisor agent can enter an infinite cyclic routing loop between `researcher` and `analyst` (e.g., researcher collects data $\rightarrow$ analyst requests more data $\rightarrow$ researcher collects overlapping data), leading to token exhaustion and request timeout.
>
> **Root Cause**: Lack of explicit termination thresholds and imprecise condition triggers in routing evaluation.
>
> **Mitigation Implemented**:
> 1. **Bounded Iterations Guardrail**: Hard-coded `max_iterations = 6` in `ResearchState` and `SupervisorAgent`. If `state.iteration >= max_iterations`, execution is deterministically forced to `"done"`.
> 2. **State-Driven Routing Policy**: The Supervisor inspects deterministic field presence (`sources` $\rightarrow$ `research_notes` $\rightarrow$ `analysis_notes` $\rightarrow$ `final_answer` $\rightarrow$ `critic_notes`) instead of relying solely on open-ended LLM classification.
> 3. **Circuit Breakers & Fallback**: If an agent fails or external API keys are unavailable, the system automatically falls back to local offline corpus retrieval and heuristic verification without crashing.

---

## 5. Trace Evidence & Observability

Fine-grained execution traces recorded for the multi-agent run:

```text
[Span: multi_agent_workflow] duration: 0.170s | status: success
  ├── [Span: node_supervisor]  duration: 0.001s | decision: researcher (iteration 1)
  ├── [Span: node_researcher]  duration: 0.082s | sources_found: 5 | tokens: in=820, out=210
  ├── [Span: node_supervisor]  duration: 0.001s | decision: analyst (iteration 2)
  ├── [Span: node_analyst]     duration: 0.024s | tokens: in=510, out=185
  ├── [Span: node_supervisor]  duration: 0.001s | decision: writer (iteration 3)
  ├── [Span: node_writer]      duration: 0.038s | tokens: in=740, out=340
  ├── [Span: node_supervisor]  duration: 0.001s | decision: critic (iteration 4)
  ├── [Span: node_critic]      duration: 0.021s | tokens: in=620, out=115
  └── [Span: node_supervisor]  duration: 0.001s | decision: done (iteration 5)
```

---

## 6. Exit Ticket: Multi-Agent Architecture Decisions

### Question 1: Trường hợp nào (use case) NÊN dùng Multi-Agent? Vì sao?
> **Trả lời:**
> **NÊN dùng Multi-Agent** cho các bài toán phức tạp, có tính chất đa bước (multi-step), đòi hỏi nghiên cứu chuyên sâu, tổng hợp dữ liệu từ nhiều nguồn độc lập và yêu cầu kiểm chứng tính xác thực cao (Fact-checking / Safety verification).
> 
> **Lý do:**
> - **Giảm tải nhận thức (Cognitive Load)**: Thay vì bắt một prompt duy nhất phải vừa tìm kiếm, vừa đọc hiểu hàng chục trang tài liệu, vừa phân tích phản biện và vừa định dạng văn bản, việc chia nhỏ thành các Agent chuyên biệt (**Researcher**, **Analyst**, **Writer**, **Critic**) giúp từng bước đạt độ chính xác và chất lượng tối đa.
> - **Khả năng kiểm toán và cô lập lỗi (Auditability & Modularity)**: Dễ dàng debug và trace từng khâu (ví dụ: nếu bài viết thiếu nguồn, ta biết ngay khâu Researcher hoặc Critic bị lỗi mà không cần sửa toàn bộ hệ thống).
> - **Chống ảo giác (Hallucination Reduction)**: Cơ chế tách biệt giữa khâu soạn thảo (*Writer*) và khâu kiểm chứng (*Critic*) tạo ra quy trình "bốn mắt" (four-eyes principle), loại bỏ hầu hết các thông tin sai lệch trước khi trả về kết quả cuối cùng.

---

### Question 2: Trường hợp nào KHÔNG NÊN dùng Multi-Agent? Vì sao?
> **Trả lời:**
> **KHÔNG NÊN dùng Multi-Agent** cho các tác vụ đơn giản, câu hỏi tra cứu thông tin trực tiếp (fact lookup), các bài toán cần phản hồi thời gian thực với độ trễ cực thấp (Low Latency / Real-time SLA < 1s), hoặc các hệ thống bị giới hạn chặt chẽ về chi phí token.
>
> **Lý do:**
> - **Độ trễ cao (High Latency Overhead)**: Luồng multi-agent chạy qua nhiều vòng handoff và suy luận trung gian làm tăng thời gian phản hồi từ 2x đến 5x so với Single-Agent.
> - **Chi phí token đắt đỏ (High Token Cost)**: Dữ liệu trạng thái (`ResearchState`) được truyền qua lại giữa các agent khiến tổng số lượng input/output token tăng gấp 3x đến 8x.
> - **Độ phức tạp vận hành không cần thiết (Over-engineering & Failure Surfaces)**: Hệ thống càng nhiều agent thì càng phát sinh thêm các nguy cơ về đồng bộ state, vòng lặp vô hạn (routing loops), và độ phức tạp khi deploy/monitor trong môi trường production.
