# Peer Review Report

## Manuscript Information
- **Title**: Keep the Clock, Contrast the Prior: Training-Free Temporal Grounding for Video LLMs under a Visual-Token Budget
- **Manuscript ID**: anonymous CVPR 2026 submission (draft `docs/paper/latex/main.tex`, sec/*.tex, fig/*.tex)
- **Review Date**: 2026-10-15
- **Review Round**: Round 1. The review also considers results produced after the draft (`recent_results.md`). The draft does not contain them.

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Perspective)

### Reviewer Identity
I work on LLM decoding and calibration: contrastive and classifier-free-guidance decoding, prior correction, and label-bias calibration such as contextual calibration and domain-conditional PMI. I have also deployed streaming egocentric and AR assistants. I am not a temporal-grounding specialist, and I defer to Reviewer 2 on benchmark conventions.

### Review Focus
1. Where PCD sits in the decoding and calibration literature, and whether its three adaptations (no-video counterfactual, time scope, confidence-adaptive α) and the plausibility cut are principled.
2. Whether "the temporal prior" is a property of the model or an artefact of the prompt format.
3. Whether the method is practical for streaming and egocentric assistants, and whether it applies beyond time tokens and beyond one model family.

---

## Overall Assessment

### Recommendation
- [x] **Major Revision**

### Confidence Score
4. Decoding and calibration are my core expertise, and the deployment questions are close to it. I am less certain about grounding-benchmark conventions.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

### Summary Assessment
The paper studies temporal grounding when a video LLM's visual KV cache is pruned to 10% of its tokens. It diagnoses two failures: QA pruners destroy timeline coverage, and a video-free "temporal prior" fills the gaps. It proposes CoCo: an equal per-frame quota filled by question attention (Coverage), plus contrastive decoding of time values against a no-video pass (PCD). It reports large gains over pruners and over the uncompressed model, measured against video-free floors.

The empirical work around PCD is careful: counterfactuals are ablated, decision rules replayed offline, negative results reported, and the floors are a welcome calibration mindset.

Two problems weaken the decoding half of the paper. First, the new results show that the "prior" depends largely on the format. On the QVHighlights-TimeLens subset, switching from the timeline prompt to a standard one-line prompt raises Coverage from 30.7 to 61.7 mIoU, and PCD then adds nothing. The QVHighlights headline is therefore measured against a prompt-handicapped baseline. Second, PCD has the same form as context-aware decoding and classifier-free guidance for LMs, and its counterfactual is the content-free input of contextual calibration. None of these are cited, and the adaptive α is not separated from overall contrast strength.

For deployment, Coverage needs the question before pruning, so it saves neither prefill nor peak memory. Random + PCD, which could be built before the question, is not presented that way. Major Revision: the decisive defect is repairable with experiments the authors largely already have.

---

## Strengths

### S1: The no-video counterfactual is chosen by ablation, not by assertion
The paper compares the empty-memory counterfactual with the counterfactual from the hallucination literature (VCD's noised frames), with permuted timestamps, with a one-sided "evidence-against" rule, with PAI-style attention gain, and with an instruction. Only the empty memory removes the early-start signature (38% → 19%). This is the right experiment for a decoding paper. It identifies what the negative distribution must contain (the prior, not corrupted content). It also explains why VCD-style negatives fail here: they still contain the prior.
**Evidence Anchor**: `table: Table tab:cf (What to contrast against) — No video +6.2* mIoU vs Noised frames +0.5 and Permuted timestamps +0.9 (n.s.)`

### S2: Video-free floors bring calibration practice into grounding evaluation
Reporting, for every method, the best constant interval and a length-rescaling baseline amounts to reporting a "content-free" reference, much as NLP calibration work compares against majority-label and content-free baselines. The finding that a constant beats the uncompressed model on three of four datasets is useful to the community regardless of the method.
**Evidence Anchor**: `table: Table tab:floors — best constant [aq,bq] beats All tokens on S-EMBER (36.9 vs 28.0), ActivityNet (37.9 vs 22.9), Charades-STA (29.7 vs 16.6)`

### S3: Restricting the contrast to time values, with the side-effects measured
Contrastive and CFG-style decoding is known to distort fluent text when applied everywhere. The authors measure this directly: changed answer text, run-on lists, and over-counting. They then restrict the contrast to time values. Scoping the guidance to a typed field is a sound idea that the decoding literature uses rarely.
**Evidence Anchor**: `text: §6.4 "Contrasting every token also changes words that have nothing to do with time: the answer text changed in 85\% of questions"`

### S4: PCD works on many memories, including streaming ones
PCD improves mIoU significantly on eight memories, among them two HERMES streaming memories and FastVID. The new results add TimeLens-8B on S-EMBER under the timeline prompt: Random + PCD reaches 38.4 vs 28.2 for Random, and CoCo 43.9 vs 33.1 unpruned. Under that format, PCD is therefore not tied to one memory or to one checkpoint.
**Evidence Anchor**: `table: Table tab:pcd8 — ΔmIoU +4.6* to +8.0* on all eight memories, early starts roughly halved`

### S5: Negative results and offline replay of decision rules
Logging both distributions and replaying alternative rules (skewed or tempered prior, evidence-against, a threshold tuned on gold labels) is good practice for decoding research and rarely seen in vision papers. The negative-results table spares readers dead ends.
**Evidence Anchor**: `table: Table tab:negative — eleven ideas with ΔmIoU against stated references, including VCD-style contrast +0.5 and evidence-against −5.6*`

---

## Weaknesses

### W1: The headline comparison uses a baseline handicapped by the prompt
**Problem**: All main-table methods share the "timeline" prompt (§5), including "All tokens". The new results show that this prompt is a poor format for single-moment grounding. On the 513-query QVHighlights-TimeLens subset, Coverage at 31% scores 30.7 mIoU under the timeline prompt and 61.7 under the standard one-line TimeLens prompt. Timeline + PCD + reserve reaches only 43.9, a paired −17.8* against the one-line prompt. Under the one-line prompt, PCD has no effect (−0.1 / −0.3, n.s.), and the unpruned Qwen3-VL-8B reaches 60.5 mIoU / 64.9 R@0.5. The draft itself admits PCD gives only +0.4 with a direct-answer prompt (§6.4), and a TODO in §4.2 says the same. The abstract nonetheless presents 28.6 → 54.8 R@0.5 as CoCo's gain over all tokens. Caveat: the 513-query subset, its resolution and its frame protocol differ from the paper's 500-query low-resolution QVHighlights copy. The absolute numbers are therefore not directly comparable, but the within-subset paired effect is clean. Scope of this Critical: it attaches to the QVHighlights headline and to §6.1(i) ("nearly doubles the uncompressed model"). The S-EMBER headline (24.6 → 40.8) is not tested by this evidence: no one-line-prompt S-EMBER baseline exists, §6.4's +0.4 is PCD's delta rather than the baseline level, and enumeration may be legitimate for counting/duration questions (see Question 1).
**Evidence Anchor**: `text: Abstract "raises R@0.5 on QVHighlights from 25.0 (random tokens) and 28.6 (all tokens)"`
**Why it matters**: If the uncompressed model under a standard prompt scores far above CoCo, the claim that CoCo "nearly doubles the uncompressed model" (§6.1(i)) says more about the prompt than about compression. A CVPR reader who knows the published Qwen3-VL QVHighlights numbers (≈59 mIoU via the TimeLens script) will notice this at once.
**Suggestion**: (a) Add a "standard prompt" block to Table tab:main: All tokens, Random, the four pruners and Coverage under the one-line prompt on all four datasets, and state the best number any configuration reaches. (b) Restate the contribution by format. Coverage is the prompt-independent result: it transfers to the one-line prompt, to TimeLens-8B and to NExT-GQA answering. PCD + reserve is a remedy for reasoning-style, enumerative outputs, which S-EMBER counting and duration questions plausibly need. (c) Move the TODO "report PCD with a direct-answer prompt" into the main text as a stated limitation.
**Severity**: Critical
**Confidence**: 4. Core expertise in prompt and format effects on decoding; relies on reported numbers from recent_results.md, not on runs I re-ran.

### W2: "The temporal prior" is presented as a model property, but the evidence points to an effect elicited by the format
**Problem**: §3.2 measures the prior under the timeline prompt: 89% of the first `Seen:` moments are at 0 s. A chronological "Seen: t₁, Seen: t₂, …" enumeration invites the model to start at the beginning of the window. "Start early" is then partly a property of listing events in order, not of a video-free belief about where events happen. §3.2 rules out an *instruction* ("do not assume early") as a remedy and concludes that the prior must be removed in the output distribution. But the instruction was tested inside the same enumerative format. The new phase-38 result shows that changing the *format* on the input side removes most of the gap. PCD also needs that format: with direct answers it does nothing (§6.4), and under the one-line prompt it is null (the authors' own run log attributes this to PCD changing only the decimal of the time, which TimeLens rounds to 1 s).
**Evidence Anchor**: `text: §3.2 "The prior acts in the output distribution and has to be removed there."`
**Why it matters**: The diagnosis is one of the paper's three contributions. If the early-start signature is largely produced by the format, the "temporal prior" is a property of the model and prompt together. Calling it a prior "that does not depend on the video" (abstract) is then too strong, and the case for a decoding-time fix weakens.
**Suggestion**: Measure the no-video signature (start-in-first-5%, end-too-early) under the one-line and direct-answer prompts on S-EMBER and QVHighlights. If the prior persists there (in seconds rather than as a "0 s" first line), the claim stands with a qualifier. If it vanishes, recast §3.2 as "an enumerative format exposes a start-of-window prior" and drop the conclusion that only decoding can remove it.
**Severity**: Major
**Confidence**: 4. Core expertise: prior and bias elicitation in LM outputs.

### W3: PCD is not placed in the decoding and calibration lineage it belongs to
**Problem**: Eq. (pcd), s = (1+α)·log p_M − α·log p_∅, has exactly the form of context-aware decoding (Shi et al., NAACL 2024). There the "context" is the video memory and the "no-context" pass is the no-video pass. It is also classifier-free guidance applied to LMs (Ho & Salimans 2022; Sanchez et al. 2023, "Stay on topic with Classifier-Free Guidance"), with guidance scale 1+α. The plausibility set 𝒱_β is the adaptive plausibility constraint of Contrastive Decoding (Li et al., ACL 2023), which reached the paper through VCD. The empty-memory counterfactual is the content-free input of contextual calibration (Zhao et al., ICML 2021, "Calibrate Before Use"). It is also the domain premise of domain-conditional PMI (Holtzman et al., EMNLP 2021, "Surface Form Competition"). Coherence boosting (Malkin et al., ACL 2022) contrasts against a truncated context in the same way. Related work cites only the vision-hallucination branch (VCD, ICD, M3ID, PAI, SEASON). The intro also lists "no labelled calibration" as a property, but contextual calibration and DC-PMI need none either.
**Evidence Anchor**: `text: §2 Contrastive decoding "These methods target object hallucination or QA."`
**Why it matters**: Without this lineage, a reader cannot tell what is new. The new parts appear to be: a time-typed scope, a confidence gate on the strength, and the observation that for timestamps the right negative is "no input" rather than "corrupted input". The alternative rules in Table tab:replay ("skewed or tempered prior", "evidence-against") are points on the same spectrum, which runs from greedy decoding (α = 0) to the PMI / calibrated argmax (α → ∞, where the ranking becomes log p_M − log p_∅). Saying so would make the replay analysis much more informative.
**Suggestion**: Add a short paragraph in §4.2 that writes PCD as CAD/CFG with guidance scale γ = 1+α and the content-free prior of Zhao et al. / Holtzman et al. Add one row to Table tab:replay or tab:cf for pure PMI (contextual-calibration-style division by p_∅, i.e. α → ∞). Recast the contribution as "which counterfactual, which tokens, and how strongly, for time values".
**Severity**: Major
**Confidence**: 5. Core expertise.

### W4: The adaptive α is not separated from the overall contrast strength, and β is inherited without a sweep
**Problem**: The adaptive rule α_t = α_max(1 − max_y p_M(y)) with α_max = 1 is compared only with a *constant α = 0.5* (Table tab:scope bottom, Table tab:pcd8 "Plain PCD"). The adaptive rule's mean α is not reported, so the +6.4* accuracy on early-evidence questions may come from a different average strength rather than from confidence gating. The rule also overlaps with the plausibility cut. When max p_M is high, 𝒱_β shrinks to a few tokens *and* α falls toward 0, so two mechanisms do the same job, and it is unclear which one is active. β = 0.1 is copied from VCD without a sensitivity check. Finally, max p_M is the confidence of one digit token, not of the time value (see W5).
**Evidence Anchor**: `equation: §4.2 Confidence-adaptive strength — α_t = α_max(1 − max_y p_M(y)), α_max = 1, compared only against constant α = 0.5 (tab:scope)`
**Why it matters**: The paper names the adaptive strength as one of three adaptations that make PCD suit grounding. If a constant α matched to the same mean strength performs as well, that adaptation is not a contribution.
**Suggestion**: (a) Report the mean and distribution of α_t under the adaptive rule, and add a constant α set to that mean. (b) Sweep β ∈ {0, 0.05, 0.1, 0.3} with adaptive and constant α, and report how often the cut, rather than α, decides a time value. (c) Optionally, gate on the entropy or margin of the *value* distribution rather than of the first digit. Entropy-gated contrastive decoding variants exist in the LM literature [to verify: specific "adaptive contrastive decoding" papers].
**Severity**: Major
**Confidence**: 4. Core expertise in contrastive decoding hyperparameters.

### W5: The contrast acts on sub-word digit tokens, not on time values
**Problem**: PCD "is applied only while a time value is written", but it scores one token at a time. With a tokenizer that splits numbers into digits (the Qwen family's tokenizers do this [to verify for Qwen3-VL]), "64" is decided as "6" then "4". So the contrast compares p_M and p_∅ over first digits, which mixes magnitude classes: "1" can begin 1, 10–19 or 100+. The recent results are consistent with this: under the one-line prompt PCD is null, and the authors' run log attributes the null to PCD changing only the decimal of the time, which TimeLens rounds to 1 s. The answer reserve repairs a second symptom: PCD rewards fine time steps until generation runs out.
**Evidence Anchor**: `text: §4.2 Time scope "PCD is applied only while a time value is written"`
**Why it matters**: A contrast at the token level over a numeric quantity is not a contrast between hypotheses about *when*. It may explain why PCD works mainly where many time values are enumerated, and not on a single interval.
**Suggestion**: Implement a value-level variant. Enumerate candidate times, for example the timestamps in the memory or a 1 s grid. Score each full string under both passes, at the cost of one batched forward per candidate set. Apply the contrast to log p(value). Compare it with the current per-digit PCD under both prompts.
**Severity**: Minor
**Confidence**: 3. Adjacent expertise in numeric tokenization; I have not inspected the tokenizer.

### W6: The method does not fit a streaming assistant, and its costs are not measured
**Problem**: The paper motivates grounding with egocentric assistants ("how long did I use the paper cutter?"). In such a system, memory must be written as frames arrive, before any question. Coverage, by design, needs the question: the question is propagated through the *full uncompressed* memory, in 3–4 scoring passes in the current implementation. Prefill, peak memory and time-to-first-token are therefore those of the uncompressed model (§7 says so). The only saving is in KV reads during decoding. On top of this, the timeline format allows up to 384 generated tokens (the cap) where a one-line answer needs roughly 15; actual output lengths are unreported. PCD runs a second, prompt-only pass for every time-value token. Neither wall-clock latency nor memory is reported anywhere; §4.1 and §7 carry TODOs.
**Evidence Anchor**: `absence: §4.1 Cost and §7 Not a streaming memory — expected measured time-to-first-token, decode latency per answer, peak memory and output length vs the uncompressed one-line-prompt model; checked §4.1, §5, §6.1–6.10, §7, appendix`
**Why it matters**: As written, the work improves accuracy given a pruned cache, not efficiency. A deployment engineer would ask why not keep the full cache and the one-line prompt, which per W1 is faster and more accurate. The 10% budget needs a stated use: decode-bandwidth savings for long multi-turn answers, multiple questions on one prefill, or memory reused across questions.
**Suggestion**: (a) Add the cost table: TTFT, tokens generated, end-to-end latency and peak memory, for All tokens (one-line prompt), All tokens (timeline), Random + PCD and CoCo. (b) Present Random + PCD and stratified or streaming-HERMES + PCD (Table tab:pcd8) as the configurations that *can* be built before the question. Random + PCD already beats the uncompressed timeline model on S-EMBER (35.0 vs 28.0). (c) If Coverage is kept, evaluate a variant written once at stream time and re-ranked at question time (the TODO in §7), or batch K questions per prefill and report cost per question.
**Severity**: Major
**Confidence**: 4. Practical experience deploying streaming assistants; numbers are not measured by me.

### W7: Generality beyond time values is untested, and the one probe was negative
**Problem**: The decomposition in Eq. (decomp) is generic. Any field the model can fill from a video-free prior (counts, durations as numbers, object locations) is a candidate for the same treatment. The only evidence the paper offers is negative: all-token PCD raised over-counting from 14% to 34%. Location questions remain unsolved (mIoU ≤ 11).
**Evidence Anchor**: `text: §6.4 "over-counting rose from 14\% to 34\% of counting questions"`
**Why it matters**: The title and framing ("Contrast the Prior") suggest a general principle. The data support it only for time values written inside an enumerative format.
**Suggestion**: Either scope the claim explicitly to time values, or add a count-scoped PCD on S-EMBER counting questions. The prior for counts is likely skewed toward small numbers, which is a separate, testable hypothesis. Report the no-video count distribution next to the no-video time distribution.
**Severity**: Minor
**Confidence**: 4. Core expertise.

### W8: Evidence comes from one model family, and the draft does not use the newer results that would partly address this
**Problem**: All results in the draft use Qwen3-VL-8B. The new results add TimeLens-8B (S-EMBER: same method ordering; NExT-GQA and QVH-TimeLens: Coverage transfers). TimeLens-8B is, as far as I know, a grounding-tuned model built on Qwen3-VL-8B [to verify]. If so, both models share the timestamp-token interface that Coverage relies on. Models that encode time differently, for example through absolute-time position encodings or frame numbers drawn on the frames, are not tested.
**Evidence Anchor**: `dataset: §5 Model — Qwen3-VL-8B-Instruct only; §7 "All results use one model"`
**Why it matters**: The mechanism ("a timestamp survives exactly when one of its frame's visual tokens survives") depends on the interface. Generalizing it requires a model whose time interface differs.
**Suggestion**: Add the TimeLens-8B results with their caveats: resolution setting and batch size. Add one non-Qwen video LLM with a different time interface, or state the interface assumption as a scope condition.
**Severity**: Minor
**Confidence**: 3. Adjacent expertise; the domain specifics belong to Reviewer 2.

### W9: The answer reserve is forced decoding, and part of PCD's gain is format repair
**Problem**: The reserve inserts "\nAnswer:" when R tokens remain. This is forced or constrained decoding. It is justified, but it means part of the "PCD" gain is recovering outputs that PCD itself broke: unparseable answers drop from 25–68 of 500 to 0, worth +1.5 to +6.1* mIoU. Under the one-line prompt the failure does not occur, and the reserve does nothing.
**Evidence Anchor**: `text: §4.2 Answer reserve "This is a format constraint, not a heuristic on the content"`
**Why it matters**: Readers should be able to separate "PCD improves the time value" from "the reserve recovers answers PCD broke". The decomposition in Table tab:reserve is a good start.
**Suggestion**: Report PCD-without-reserve restricted to the questions that parse under both conditions. Mention grammar-constrained decoding as the general tool. A grammar for `Time: [s, e]` that forbids unbounded `Seen:` lists would make the reserve unnecessary.
**Severity**: Minor
**Confidence**: 4. Core expertise in constrained decoding.

### W10: "PCD is near the limit of decoding" rests on a replay of the first time value only
**Problem**: The replay in Table tab:replay scores only the *first* time value, on one memory (random 10%), and only on the start/leak trade-off. The paper's own diagnosis is that the end is the weak boundary. Some rules (pure PMI, value-level contrast, joint (s, e) decoding) are not in the replay.
**Evidence Anchor**: `text: §6.4 "Even a threshold tuned on the gold labels cannot beat PCD's trade-off"`
**Why it matters**: This sentence could discourage follow-up work on decoding rules for the end boundary, which is where the paper's own analysis puts most of the remaining error.
**Suggestion**: Restrict the claim to "the start of the interval", or extend the replay to the end value. Add PMI and value-level rules.
**Severity**: Minor
**Confidence**: 4. Core expertise.

---

## Detailed Comments

### Assumption Audit
- **Explicit assumptions**: (i) The video-free distribution p_∅ is the "prior" that leaks into p_M. This holds only if the information in the memory and the prior combine roughly as in Eq. (decomp). That is a log-linear product-of-experts assumption, standard in CAD/CFG, but it should be stated. (ii) Confidence of p_M at a digit token stands for confidence about time (W4, W5). (iii) An equal per-frame quota is the right coverage constraint. This is well tested in Table tab:design.
- **Implicit assumptions**: (i) The timeline prompt is a neutral choice shared by all methods. The new results show it is not neutral: it creates the phenomenon PCD repairs (W1, W2). (ii) A per-question rebuild of memory is an acceptable deployment model (W6). (iii) The conditional p_∅(y | x, y_<t) remains a "prior" late in the answer. In fact, by the time the interval is written, y_<t contains `Seen:` times decoded from the memory. p_∅ is then a text-only continuation that already carries video information through the prefix, so the contrast at the interval is between "memory + prefix" and "prefix only". This is consistent with CAD-style practice, but it differs from the no-video prior measured in §3.2, and the difference should be acknowledged.
- **Paradigmatic assumptions**: Compression is evaluated by accuracy at a token budget, not by cost at an accuracy target. In systems work the natural question is the reverse: what is the cheapest configuration that reaches the uncompressed model's accuracy under its best prompt?

### Cross-Disciplinary Connections
- **Parallel research**: The calibration line (Zhao et al. 2021; Holtzman et al. 2021; Fei et al. 2023 on domain-context calibration [to verify venue]) faced the same problem: a content-free prior biases the output toward certain labels. Its tools are division by the content-free probability, and PMI scoring of whole answers rather than tokens. Both carry over directly to PCD (W3, W5). The CAD and CFG-for-LM line (Shi et al. 2024; Sanchez et al. 2023) has studied guidance strength and degeneration at high guidance, which matches the run-on `Seen:` lists the reserve fixes.
- **Borrowing opportunities**: (i) Value-level PMI over candidate times (W5). (ii) Grammar-constrained decoding for the interval (W9). (iii) Calibration diagnostics: a reliability diagram for the start-in-first-5% decision, with greedy vs PCD, would show directly whether PCD *calibrates* or merely *shifts* the start distribution.
- **Methodological borrowing**: CFG work reports a curve of guidance scale against quality. A curve of α_max (or γ = 1+α) against mIoU, early-start rate and parse failures, under both prompts, would replace several single-point tables.

### Practical Impact
- **Real-world application**: The most transferable findings are Coverage (QA pruners destroy timelines; keep a few tokens per frame) and the floors. PCD + timeline is a remedy for a format a deployment would probably not choose for single-moment queries. For multi-event grounded QA (counting, durations), an enumerative format is reasonable, and PCD then helps. The paper should state this condition.
- **Implementation feasibility**: Running the no-video pass alongside the main pass is cheap in memory (prompt-only cache) but doubles forward calls on time tokens, and the timeline format allows outputs up to ~25× longer (384-token cap vs ~15; actual lengths unreported). Coverage blocks memory written once before the question (W6).
- **Stakeholders**: Egocentric memory is personal data. The paper's design keeps the full raw cache until the question arrives, so memory compression gives no data-minimisation benefit. A brief note would be enough.

### Broader Implications
- **Ethical dimensions**: Beyond the privacy note above, none specific.
- **Social impact**: The floors finding (constants beat the model) is a useful warning against overstated grounding claims in the wider field.
- **Future directions**: (i) A decoder that decides open vs closed intervals (already flagged in the appendix). (ii) Value-level, end-aware contrastive decoding. (iii) Coverage with a query-agnostic score that can be computed when frames are written, re-ranked with the question at question time.

### Title & Abstract
The subtitle "Contrast the Prior" and the abstract's phrase "a prior that does not depend on the video" should be qualified by format (W2). The headline comparison should include the standard-prompt baseline (W1).

---

## Cross-Disciplinary Reading Recommendations
- Zhao, Wallace, Feng, Klein, Singh (2021). *Calibrate Before Use: Improving Few-Shot Performance of Language Models.* ICML. Content-free input as a prior estimator; the closest antecedent of the no-video counterfactual.
- Holtzman, West, Shwartz, Choi, Zettlemoyer (2021). *Surface Form Competition: Why the Highest Probability Answer Isn't Always Right.* EMNLP. Domain-conditional PMI, i.e. PCD at α → ∞ with a domain premise as the negative.
- Li et al. (2023). *Contrastive Decoding: Open-ended Text Generation as Optimization.* ACL. Source of the adaptive plausibility constraint (β).
- Shi et al. (2024). *Trusting Your Evidence: Hallucinate Less with Context-aware Decoding.* NAACL. The same (1+α)/−α form as Eq. (pcd).
- Ho & Salimans (2022). *Classifier-Free Diffusion Guidance* (arXiv). Sanchez et al. (2023). *Stay on Topic with Classifier-Free Guidance* (arXiv; later venue [to verify]). Guidance-scale framing and degeneration at high scale.
- Malkin, Wang, Jojic (2022). *Coherence Boosting: When Your Pretrained Language Model Is Not Paying Enough Attention.* ACL. Contrast against a truncated context.
- O'Brien & Lewis (2023). *Contrastive Decoding Improves Reasoning in Large Language Models* (arXiv). Contrastive decoding applied to structured, reasoning-style outputs.
- Xiao et al. (2024). *Efficient Streaming Language Models with Attention Sinks.* ICLR. Relevant to the attention-sink effect on attention-mass quotas and to streaming KV design.
- Chen et al. (2024). *VideoLLM-online: Online Video Large Language Model for Streaming Video.* CVPR. A streaming-assistant reference point for the deployment framing.
- [UNVERIFIED search lead] Entropy- or confidence-adaptive contrastive decoding for LLMs, as a precedent for the α_t gating.
- [UNVERIFIED search lead] Tokenization of numbers and its effect on numeric generation (e.g. work on digit-level vs multi-digit tokenization in arithmetic), relevant to W5.
- [UNVERIFIED] M3ID's (cited) weighting schedule over generated tokens. If it grows with generation length, it is a direct counterpart to the confidence-adaptive α and should be contrasted explicitly.

---

## Questions for Authors
1. Under the one-line prompt on S-EMBER and QVHighlights, what are the no-video early-start rate and the end-too-early rate? Does the "temporal prior" exist outside the enumerative format?
2. What is the best grounding number any configuration reaches on each dataset under the standard prompt (unpruned, Coverage at 10%), and how does CoCo-with-timeline compare to it?
3. What is the mean α_t under the adaptive rule, and does a constant α at that value match the adaptive rule on the 78 early-evidence questions?
4. In which deployment does a cache pruned to 10% after a full-cache prefill pay off: multi-turn reuse, several questions per prefill, or decode-bandwidth limits? Can you quantify the saving?

---

## Minor Issues

### Language / Clarity
- §4.2: state that p_∅ conditions on the generated prefix, which already contains times decoded from the memory, so "prior" at the interval is a prefix-conditioned text-only distribution.
- §1: "no labelled calibration" should not be listed as a distinguishing property, since content-free calibration also needs no labels.

### Figures and Tables
- Fig. overview (c): the probabilities are illustrative. Say so in the caption, not only in the source comment.
- Table tab:scope bottom: list the α value or schedule in every row.

### Citation Format
- Add the CAD/CFG/calibration references above to §2 "Contrastive decoding".

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Card #4 focus 1 (PCD vs PMI/CFG/calibration) | PARTLY_MEETS | text: §2 "These methods target object hallucination or QA." | PCD has the form of CAD/CFG with a content-free prior. What is new is the time scope, the choice of counterfactual and the gating, and these are under-positioned. | Novelty of Coverage is outside my remit. | yes: repairable through positioning |
| Methodological Rigor | Card #4 focus 1 (adaptive α principled?) | PARTLY_MEETS | equation: §4.2 α_t rule vs constant α = 0.5 | Strength is not matched to the gating; β is not swept; the contrast works per token on digits. | Statistical design is R1's remit. | yes: moderate |
| Evidence Sufficiency | Card #4 "model property vs prompt artefact" | DOES_NOT_MEET | table: recent_results §3c / phase 38: timeline + PCD vs one-line prompt −17.8* | The headline is measured against a baseline handicapped by the prompt; PCD is null under the standard prompt. | Evidence is from post-draft runs on a different QVH subset and resolution. | yes: decisive, repairable |
| Argument Coherence | Card #4 focus 1–2 | PARTLY_MEETS | text: §3.2 "The prior acts in the output distribution and has to be removed there." | The decoding-only conclusion does not follow once the format is a variable. | Internal-consistency auditing is the DA's remit. | yes |
| Writing Quality | General | MEETS | — | Clear and well organized; TODOs remain. | Not my focus. | no |
| Literature Integration | Card #4 focus 1 | PARTLY_MEETS | text: §2 Contrastive decoding paragraph (as above) | The NLP calibration and guidance lineage is missing. | Systematic coverage is R2's remit. | yes: repairable |
| Significance & Impact | Card #4 focus 2–3 (streaming, generality) | PARTLY_MEETS | absence: §4.1/§7 cost (W6) | Coverage and the floors transfer. PCD is specific to a format and to time values. No streaming saving is shown. | Costs not measured by me. | yes |

The recommendation (Major Revision) rests mainly on one repairable defect: the headline comparison uses a prompt-handicapped baseline (W1). The supporting issues are the format-dependent framing of the prior (W2), the missing decoding and calibration lineage (W3), the unseparated adaptive α (W4), and the missing cost and streaming analysis (W6). Strengths on other criteria do not offset W1. Fixing W1 needs re-baselining and reframing, not new methods. Much of the evidence already exists in the authors' newer runs.
