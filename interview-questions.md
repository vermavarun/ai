# AI Engineer Interview Questions (Tough)
*Curated question bank for top-tier AI/ML interviews*

---

## 1. Mathematics & Fundamentals

- [ ] 1.1 Derive gradient descent update for linear regression from first principles.
- [ ] 1.2 Explain why $L_2$ regularization is equivalent to a Gaussian prior.
- [ ] 1.3 Compare MLE vs MAP with a concrete probabilistic model.
- [ ] 1.4 Derive bias-variance decomposition for squared error.
- [ ] 1.5 Why do eigenvectors matter in PCA and what fails when covariance is ill-conditioned?
- [ ] 1.6 Show how SVD can be used for low-rank approximation and denoising.
- [ ] 1.7 Explain KL divergence asymmetry and where it causes practical issues.
- [ ] 1.8 When would Jensen's inequality appear in ML optimization proofs?
- [ ] 1.9 Explain concentration inequalities and their relevance to generalization.
- [ ] 1.10 How do you reason about sample complexity in high-dimensional settings?

## 2. Classical Machine Learning

- [ ] 2.1 Why can logistic regression still outperform deep models on tabular data?
- [ ] 2.2 Compare bagging vs boosting with failure modes of each.
- [ ] 2.3 Explain calibration vs discrimination in classifiers.
- [ ] 2.4 How would you debug a model with high AUC but poor business outcomes?
- [ ] 2.5 When does cross-validation give misleading confidence?
- [ ] 2.6 Explain data leakage patterns that are hard to detect.
- [ ] 2.7 How do you design a robust feature store for online/offline parity?
- [ ] 2.8 Compare SHAP, permutation importance, and partial dependence limitations.
- [ ] 2.9 What are the assumptions behind Naive Bayes and when does it still work well?
- [ ] 2.10 How do you evaluate models under severe class imbalance?

## 3. Deep Learning Core

- [ ] 3.1 Explain vanishing/exploding gradients with Jacobian products.
- [ ] 3.2 Why does BatchNorm help optimization beyond internal covariate shift arguments?
- [ ] 3.3 Compare AdamW vs Adam and explain decoupled weight decay.
- [ ] 3.4 How would you choose learning rate schedules for large-batch training?
- [ ] 3.5 Explain label smoothing and when it hurts.
- [ ] 3.6 Derive attention complexity and practical memory bottlenecks.
- [ ] 3.7 Why can deeper models generalize better despite overparameterization?
- [ ] 3.8 How do residual connections improve optimization landscapes?
- [ ] 3.9 What is sharpness-aware minimization (SAM) and when is it useful?
- [ ] 3.10 Explain activation checkpointing trade-offs in training large models.

## 4. Transformers & LLM Internals

- [ ] 4.1 Walk through a token from input to next-token probability.
- [ ] 4.2 Compare BPE, WordPiece, and SentencePiece impacts on downstream behavior.
- [ ] 4.3 Explain rotary positional embeddings and why they improved long-context behavior.
- [ ] 4.4 What are grouped-query and multi-query attention, and why do they matter for inference?
- [ ] 4.5 Explain KV cache design and eviction policies for long sessions.
- [ ] 4.6 Compare full fine-tuning, LoRA, QLoRA, and adapter tuning.
- [ ] 4.7 How do you design SFT datasets to reduce catastrophic forgetting?
- [ ] 4.8 Explain DPO vs RLHF and when each is preferable.
- [ ] 4.9 How do reasoning-tuned models differ from instruction-tuned models?
- [ ] 4.10 What breaks when you aggressively quantize a model?

## 5. RAG and Retrieval Systems

- [ ] 5.1 Design a production RAG system for 100M documents and strict latency SLOs.
- [ ] 5.2 Compare dense retrieval, sparse retrieval, and hybrid retrieval in failure cases.
- [ ] 5.3 How do you choose chunk size and overlap scientifically?
- [ ] 5.4 Explain reranking models and where to place them in the pipeline.
- [ ] 5.5 How would you evaluate retrieval quality separately from generation quality?
- [ ] 5.6 How do you enforce access control in retrieval for multi-tenant systems?
- [ ] 5.7 Describe retrieval drift and how to detect it in production.
- [ ] 5.8 How do you handle freshness vs relevance trade-offs?
- [ ] 5.9 Explain citation hallucinations and mitigation approaches.
- [ ] 5.10 Design fallback strategies when retrieval fails.

## 6. AI Agents and Tool Use

- [ ] 6.1 Design an agent architecture for reliable tool execution under uncertainty.
- [ ] 6.2 What are common failure loops in multi-step agents and how do you break them?
- [ ] 6.3 Compare planner-executor, ReAct, and graph-based orchestration.
- [ ] 6.4 How do you evaluate an agent beyond "task success"?
- [ ] 6.5 How do you make tool calling deterministic and auditable?
- [ ] 6.6 Explain memory design for agents (episodic, semantic, working memory).
- [ ] 6.7 How do you sandbox tool execution for security?
- [ ] 6.8 How do you version prompts, tools, and policies safely?
- [ ] 6.9 What does a strong rollback strategy look like for agent deployments?
- [ ] 6.10 How would you run red-team tests against an agent system?

## 7. MLOps, Deployment, and Reliability

- [ ] 7.1 Design CI/CD for models with data + code + prompt versioning.
- [ ] 7.2 Compare canary, shadow, and blue/green for model rollout.
- [ ] 7.3 What metrics do you monitor for LLM apps in production?
- [ ] 7.4 How do you detect silent quality regressions after a model update?
- [ ] 7.5 Explain inference autoscaling strategies for GPU fleets.
- [ ] 7.6 How do you optimize cost per successful task, not just tokens?
- [ ] 7.7 What belongs in an AI incident postmortem?
- [ ] 7.8 How do you establish SLOs for generative systems with stochastic outputs?
- [ ] 7.9 How do you test reproducibility in distributed training jobs?
- [ ] 7.10 How do you handle schema drift in upstream data pipelines?

## 8. Responsible AI, Security, and Privacy

- [ ] 8.1 Explain prompt injection attacks and layered defenses.
- [ ] 8.2 How do you prevent model output from leaking secrets?
- [ ] 8.3 Describe privacy controls for chat logs containing sensitive data.
- [ ] 8.4 What is model inversion and when is it a realistic threat?
- [ ] 8.5 How would you evaluate fairness in an LLM-enabled decision workflow?
- [ ] 8.6 Design policy enforcement for high-risk tool actions.
- [ ] 8.7 How do you do safety evals for multilingual applications?
- [ ] 8.8 What are trade-offs between strict filtering and user utility?
- [ ] 8.9 How do you audit model behavior over time for compliance?
- [ ] 8.10 How do you build human-in-the-loop escalation paths?

## 9. System Design Scenarios

- [ ] 9.1 Design an enterprise copiloting system for code, docs, and tickets.
- [ ] 9.2 Design a multimodal assistant (text+image+audio) with real-time constraints.
- [ ] 9.3 Design a customer support AI with guaranteed citation grounding.
- [ ] 9.4 Design a long-context legal-document assistant with strict privacy boundaries.
- [ ] 9.5 Design a fraud detection stack with both classic ML and LLM explanations.
- [ ] 9.6 Design experimentation for prompt/model/retrieval changes simultaneously.
- [ ] 9.7 Design a model routing layer across small/large/specialist models.
- [ ] 9.8 Design cost controls and budget enforcement for API-heavy AI systems.
- [ ] 9.9 Design a resilient architecture for regional outages.
- [ ] 9.10 Design observability that ties latency, quality, and business metrics.

## 10. Debugging and Practical Case Questions

- [ ] 10.1 Your RAG system is fast but wrong. How do you isolate root cause?
- [ ] 10.2 Online metrics dropped after quantization; what checks do you run first?
- [ ] 10.3 Agent succeeds in staging but fails in production intermittently; how do you diagnose?
- [ ] 10.4 Hallucinations rose after adding new documents; what likely changed?
- [ ] 10.5 Model latency doubled without code changes; what infra signals do you inspect?
- [ ] 10.6 Training loss decreases but eval loss increases; what hypotheses do you test?
- [ ] 10.7 A/B test says prompt B is better, users complain quality is worse; why?
- [ ] 10.8 Your model performs poorly for one customer segment; how do you investigate fairness vs data skew?
- [ ] 10.9 GPU utilization is low but queue is high; where is the bottleneck likely?
- [ ] 10.10 Grounded answers dropped after retriever update; how do you roll back safely?

## 11. Behavioral + Leadership (Senior/Staff)

- [ ] 11.1 Describe a high-impact AI system you shipped and its measurable outcomes.
- [ ] 11.2 Tell me about a major AI failure you handled and what changed afterward.
- [ ] 11.3 How do you decide when not to use an LLM?
- [ ] 11.4 How do you influence product decisions with uncertainty in model quality?
- [ ] 11.5 How do you mentor junior engineers on experiment rigor?
- [ ] 11.6 Describe a disagreement with research/product and how you resolved it.
- [ ] 11.7 How do you prioritize reliability vs velocity in AI product roadmaps?
- [ ] 11.8 How do you communicate model risk to non-technical stakeholders?
- [ ] 11.9 How do you define engineering excellence for AI teams?
- [ ] 11.10 What is your framework for technical decision-making under ambiguity?

---

## 12. How to Practice These Questions

- [ ] 12.1 Answer each question in 3 layers: intuition, math/system details, production trade-offs.
- [ ] 12.2 For every system design answer, state assumptions, constraints, and failure modes.
- [ ] 12.3 Add one metric and one rollback strategy to every architecture answer.
- [ ] 12.4 Build a personal answer bank with diagrams and postmortem examples.
- [ ] 12.5 Run timed mock interviews: 45 min system design + 30 min deep dive.

---

*Last Updated: June 28, 2026*
