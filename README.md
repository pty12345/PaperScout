# PaperScout: An Autonomous Agent for Academic Paper Search with Process-Aware Sequence-Level Policy Optimization

> 🔍 PaperScout is an autonomous LLM-based agent that reformulates academic paper search as a **multi-turn decision-making process**, dynamically deciding *when* and *how* to invoke search and citation expansion tools.
> 
> 🧠 To train such agents effectively, we introduce **PSPO (Proximal Sequence Policy Optimization)**, a process-aware, sequence-level RL algorithm tailored for agentic retrieval.

📄 **Paper**: [https://arxiv.org/pdf/2601.10029](https://arxiv.org/pdf/2601.10029).

## 🚀 From Static Retrieval to Agentic Search

Academic paper search is a core step in scientific research, but existing systems suffer from non-negligible limitations.

> **Semantic Match:**
> Treats paper search as a single-shot retrieval problem, assuming relevance can be determined from the query alone.
> 
> **Fixed Workflow:**
> Decomposes search into multiple steps, but follows a predefined pipeline that cannot adapt to evolving search results.

PaperScout addresses these challenges by **treating paper search as a sequential decision-making problem**, enabling flexible, context-aware automatic paper discovery.

<!-- IMAGE PLACEHOLDER -->

<!-- Suggested image: Figure 1 from the paper -->

<!-- Path suggestion: assets/overview_paradigms.png -->

<!-- <img width="70%" height="651" alt="image" src="https://github.com/user-attachments/assets/63b17a39-e15c-4f8b-b0b6-8d3849296f51" /> -->

---

## 🧠 PaperScout Framework

PaperScout models academic paper search as a **POMDP**, where the agent iteratively interacts with an external retrieval environment.

<!-- IMAGE PLACEHOLDER -->

<!-- Suggested image: Figure 2 (left) -->

<img width="1659" height="634" alt="image" src="https://github.com/user-attachments/assets/087f7fe6-0cd8-4703-95ec-def587d97f10" />


### Core Components

* **State.**
  A latent *paper pool* containing all retrieved papers so far.

* **Observation.**
  A summarized view of the pool, including:

  * top-ranked expanded papers,
  * candidate papers for further expansion,
  * interaction history to avoid redundancy.

* **Actions.**

  * `Search(query)`: retrieve new papers from scholarly search engines
  * `Expand(paper)`: follow citation links of an existing paper

* **Reward.**
  Designed to maximize **expected recall of relevant papers**, while penalizing redundant actions.

---

## ⚙️ Training Challenge: Why PSPO?

Training multi-turn retrieval agents with standard RL methods is surprisingly hard:

* **Token-level PPO**
  ❌ Misaligned with agent turns → noisy credit assignment
* **Outcome-only methods (e.g., GRPO)**
  ❌ Too coarse → ignore intermediate process signals

### ✨ Our Solution: PSPO

**Proximal Sequence Policy Optimization (PSPO)**, which aligns optimization *exactly* with the agent’s interaction granularity.

<!-- IMAGE PLACEHOLDER -->

<!-- Suggested image: Figure 2 (right) -->

**Key properties of PSPO:**

* Treats each *complete agent response* as an atomic action
* Performs **sequence-level advantage estimation**
* Incorporates **process rewards** via a learned critic
* Significantly improves **training stability and sample efficiency**

---

## 📊 Experimental Results

PaperScout is evaluated on both **synthetic** and **real-world** academic search benchmarks.

### Main Results

* Consistently outperforms:

  * Google Search / Google Scholar
  * Workflow-driven agents (PaSa, SPAR)
* Achieves **higher recall with fewer tool calls**
* A **4B model trained with PSPO** matches or surpasses a much larger untrained backbone

<!-- IMAGE PLACEHOLDER -->

<!-- Suggested image: Table 2 or Figure 3 -->

<img width="1240" height="468" alt="image" src="https://github.com/user-attachments/assets/eee01fd8-5d33-4353-acbc-875a013dbfdf" />


---

## 🔎 Case Study: Adaptive Multi-Turn Retrieval

PaperScout dynamically alternates between *Search* and *Expand* as retrieval progresses.

<!-- IMAGE PLACEHOLDER -->

<!-- Suggested image: Figure 6 -->

<img width="1764" height="435" alt="image" src="https://github.com/user-attachments/assets/b7f63386-f3ae-4abf-b846-e66f4cbf68fa" />


**Observation:**
When citation expansion becomes saturated, the agent *re-initiates search* to explore new directions—behavior that fixed workflows cannot express.

---

## 🧩 Key Contributions

* 🧠 **PaperScout**: the first autonomous paper search agent with fully adaptive retrieval decisions
* ⚙️ **PSPO**: a process-aware, sequence-level RL algorithm for multi-turn agents
* 📈 Strong empirical gains in recall, relevance, and training stability

---

## 📌 Citation

```bibtex
@article{pan2026paperscout,
  title={PaperScout: An Autonomous Agent for Academic Paper Search with Process-Aware Sequence-Level Policy Optimization},
  author={Pan, Tingyue and Ouyang, Jie and Cheng, Mingyue and Li, Qingchuan and Liu, Zirui and Pan, Mingfan and Yu, Shuo and Liu, Qi},
  journal={arXiv preprint arXiv:2601.10029},
  year={2026}
}
```

