# 03 Methods: Letting the Reader Know What Was Done, Why It Was Done That Way, and How to Reproduce It

## 1. Section Mission and Boundary

The primary function of the Methods is to provide enough information for other researchers to reproduce the work and obtain similar results. The Methods may be called Methodology, Experimental, Procedure, Model, Data Collection etc., and may also be placed in supplementary materials; the name and location may vary, but the reader's need for choices, steps, parameters, materials, equipment and problems does not disappear.

**Systems-paper mapping (governs this project).** A systems-conference paper has no section called "Methods". The functions of this guideline distribute as follows (structure per guideline 14):

- **Design sections (2–3)** carry M1/M2/M4/M5/M6: the mechanism, the rationale for choices, the relationship to existing techniques, and honest costs/limits. Each design section is insight-first (SYS-05).
- **Implementation** carries the reproducibility function: stack, LOC, built-vs-reused, non-obvious engineering (SYS-06); full reproducibility is delivered by the released artifact.
- **Evaluation §Setup** carries the experimental-procedure function: hardware, workloads, baselines, metrics, statistics protocol (SYS-07).
- The rules below about sequence language, rationale, ownership and honest problems apply unchanged to design and setup text. Rules written for wet-lab procedure (equipment description, spatial language) apply only when actually describing physical apparatus.

## 2. The Six-Component Model

```text
M1 Method overview / aim / source of materials or equipment
M2 Specific materials, parameters, steps and order; rationale and careful handling
M3 Description of a figure, table, model or equipment
M4 Same as, similar to, or significantly different from existing methods
M5 Present Simple background facts, used to support reader understanding or choice rationale
M6 Problems and limitations in the method, and their impact
```

This is a menu, not all of which must be used. Whether to place M3, M4 and M6 in the main text should be decided by the type of research, the target journal and the reader's needs.

## 3. Rules

### METH-01 Go from Overview to Detail, Give the "Wall" Before the "Bricks"

- Rule strength: NORMALLY.
- Writing action: First state what the research does, what it uses, where the samples come from and what the aim is, then expand step by step.
- Sentence/paragraph skeleton: `The current investigation involved [activity] to determine [aim].` / `A three-step approach was used to [purpose].`
- Conditions and variants: When the readership is highly specialised and the method needs no explanation, you may start directly with the detail; interdisciplinary or information-surfing readers need an overview.
- Why: If the reader assembles the method upwards from the detail, different readers will construct different overall understandings.
- Completion criterion: The reader can state the object, purpose and general route of the method before entering the second sentence.
- Source: Unit 2 §2.1–2.2.2.

### METH-02 Write Reproducible Steps, Not a Lab Log

- Rule strength: MUST.
- Writing action: Write material sources, equipment models, parameters, quantities, temperatures, times, order, stopping conditions, replication counts and analysis procedures; use precise sequence language instead of rough connectors such as only then/next. In a systems paper the same rule reads: exact hardware/software versions, dataset/workload identity, algorithm parameters and defaults, protocol step order with trigger conditions, seeds/repetitions, and the boundary at which each measurement starts and stops.
- Sentence/paragraph skeleton: `Prior to [event], [operation] was performed using [equipment]. [Measurement] was monitored until [condition], at which point [next operation].`
- **The eight groups of sequence language** (choose by meaning, don't use only then/next):

| Group | Function | Common words |
|---|---|---|
| 1. Before the experiment | Events that occurred before beginning | beforehand, earlier, formerly, in advance, originally, previously, prior to |
| 2. Beginning | The first step of the experiment or observation | at first, at the beginning, at the start, firstly, in the beginning, initially, to begin with, to start with |
| 3. Order | Tells the order but not the time interval | after, afterwards, followed by, following, next, secondly, subsequently, then |
| 4. Short interval | Only a short time between two events | quickly, shortly after, soon |
| 5. Long interval/later stage | After a longer time or near the end | eventually, in due course, in time, later, later on, subsequently, towards the end |
| 6. Simultaneous | Two events occur at the same time or interface seamlessly | as, as soon as, at once, at the same time, immediately, instantly, in the meantime, meanwhile, once, simultaneously, straight away, until, when, while |
| 7. End | The last step of the sequence | at the end, eventually, finally, in the end, lastly |
| 8. After the experiment | Events that occurred after the experiment ended | afterwards, eventually, later, later on, subsequently |

- Key point: `then` and `next` only tell the order, not the time interval. If waiting for stability or the time interval affects the result, you must write the condition using until, once, at which point, shortly after etc.
- Completion criterion: Another researcher can reconstruct the timeline from the text and knows the role of every parameter.
- Source: Unit 2 §2.2.2, §2.4.2.

### METH-02b Use Spatial Language to Let the Reader "See" the Equipment

- Rule strength: CONDITIONAL (only when describing physical self-built equipment or complex apparatus; **not applicable to software-systems papers** — for describing software architecture, use the component/interface/dataflow conventions of guideline 14 SYS-04 instead).
- Writing action: Use accurate spatial prepositions and verbs to describe the position, orientation and connection of equipment components, so that the reader can visualise the apparatus.
- Common spatial prepositions: above, adjacent, across, along, against, below, under, beside, alongside, downstream/upstream, inside/within, parallel to/perpendicular to, on each/either/both side, opposite/facing.
- Common spatial verbs: align, arrange, assemble, attach to, connect, couple, embed, encase, enclose, fasten, fit, fix, install, intersect, join, locate, mount, orient, place, position, situate, space, surround.
- Precision words: just above, slightly above, immediately above, directly above, right above — the meanings differ and they are not interchangeable.
- Completion criterion: The reader can reconstruct the spatial layout of the equipment in their mind, without needing to see a photograph.
- Source: Unit 2 §2.4.2.

### METH-03 Provide a Rationale for Key Methodological Choices

- Rule strength: NORMALLY.
- Writing action: State why a particular material, equipment, model, parameter or control was chosen; give the precision, sensitivity, stability, simplification or control advantage it brings.
- Sentence/paragraph skeleton: `[Material/equipment] was selected in order to [function].` / `This allowed us to [benefit].` / `The choice was made because [property].`
- Why: The degree to which the reader accepts the results depends on whether they accept the method; "that's just how I did it" is no substitute for a reason.
- Common errors -> Fix action: `samples were stored` -> `samples were stored at [condition] to prevent [risk] until [analysis]`.
- Completion criterion: Every choice that a reviewer might ask "why" about has a rationale or literature basis next to it.
- Source: Unit 2 §2.2.2, §2.4.2.

### METH-04 Show Care and Quality Control, but Don't Write Self-Praise

- Rule strength: NORMALLY.
- Writing action: Accurately use observable adverbs such as carefully, directly, separately, tightly, thoroughly, repeatedly, independently, randomly, and explain how the control affects reliability.
- Common errors: Using "carefully" without describing the actual operation; using exaggerated adjectives in place of parameters and controls.
- Completion criterion: Every quality judgement can be traced back to an operation, a control, a replication or an equipment characteristic.
- Source: Unit 2 §2.2.2, §2.4.2.

### METH-05 Make the Relationship to Existing Methods Explicit

- Rule strength: MUST when reusing a method; otherwise CONDITIONAL.
- Three ways of expressing it:
  - Exactly the same: `as described by/in...`, `using the method of...`, `identical to...`;
  - Similar or modified: `adapted from`, `a modified version of`, `with some adjustments`;
  - Significantly different: state what the change was, why it was made and what it improved, using a contrast signal if necessary.
- Citation location: immediately follow the method that actually comes from others, so the reader does not mistakenly think the authors of this paper originated the method.
- Completion criterion: The reader can distinguish standard procedures, others' methods and the parts added or changed in this paper.
- Source: Unit 2 §2.2.2, §2.4.2, §2.5.1.

### METH-06 Active/Passive and Tense Must Serve Ownership

- Rule strength: MUST.
- The Past Simple agentless passive normally reports what was done in this study; the Present Simple agentless passive normally describes standard procedures, equipment functions or stable facts.
- Dangerous contrast: `A flexible section was inserted...` (done in this study) versus `A flexible section is inserted...` (standard/usual practice).
- Fix action: Use `In this study`, `here`, `in our model`, `according to their study`, `using standard procedures`, or switch to `We...`.
- Why: The agentless passive looks identical in form, so the reader may mix together this paper's work, others' work and standard knowledge. Remember the key message: The aim is not simply to make it possible for the reader to understand; the aim is to make it impossible for the reader NOT to understand.
- Completion criterion: Go through sentence by sentence asking "who did this action, when, and is it this paper's contribution or background"; the answer does not depend on the author's intuition of familiarity with the project.
- Source: Unit 2 §2.5.1; Unit 8 §8.3.

### METH-07 Write Problems Honestly, and Control Their Significance

- Rule strength: MUST when the problem may affect interpretation.
- Writing action: State the problem first in the Methods paragraph where it occurs; minimise the severity of the problem and your responsibility, maximise the part that is still reliable, and give a reason or a direction for future repair.
- Sentence/paragraph skeleton: `Although [issue], the effect on [outcome] was negligible.` / `The procedure was slightly problematic in that [specific problem].`
- Not to do: Leaving the limitation to be mentioned for the first time in the Conclusion, or pretending the method was perfect; this makes it look as though the authors are unaware of the problem.
- Completion criterion: The reader knows the problem, the extent of the impact, why the current results are still interpretable, and how to improve subsequently.
- Source: Unit 2 §2.2.2, §2.4.2.

## 4. Practical Control of Prepositions and Articles

- `using` is more suitable for indicating a tool or equipment; `by + -ing` is more suitable for indicating the process by which a result was achieved.
- `evidence of` indicates a measurable sign that is already there; `evidence for` indicates support for something that may exist.
- `substituted for` and `substituted with` differ in the direction of substitution, and cannot be interchanged by habit.
- Avoid stacking prepositional phrases in succession; split the sentence when necessary and turn time, place and purpose into explicit content words.
- `a/an` can be used for first introduction, uncertainty or a general singular; the zero article is used for plural general statements or uncountable concepts; `the` indicates uniqueness, known-ness or shared knowledge the reader can infer.

## 5. Methods Paragraph Blueprint

1. Overview: aim, sample/equipment source and overall scheme;
2. Design: grouping, controls, model or experimental system;
3. Operation: write the steps in temporal and logical order;
4. Explanation: insert choice rationale, advantages and relationship to existing methods where needed;
5. Analysis: state the measurements, calculations, criteria and replications;
6. Problems: state limitations, impact and feasible follow-up handling.

## 6. Methods Completion Checklist

- Does the reader know the overall aim and route of the method?
- Can every step be reproduced, without having to guess the timing of then/next?
- Are materials, equipment, parameters and sample sources sufficiently explicit?
- Do key choices have a rationale?
- Are this paper's work, others' methods and standard procedures separated?
- Are the positions of figures, tables, models and equipment explained?
- Do problems appear where they occur, rather than suddenly at the end?
- Can supplementary materials be located via specific pointers in the main text?


## Worked Examples from the Book: Whether the Methods Are Sufficient to Reproduce

### Counter-example: Lab notebook

> An Ag wire was attached to the electrode. The cathode was sealed. The sample was tested.

### Good example: Action, purpose and order are all traceable

> An Ag wire was attached to the electrode with Ag paste in order to form an electrical contact. The cathode was then sealed to prevent leakage before the sample was tested.

When checking the good example, circle the material, the action, the purpose, the order and the risk control separately. The Methods are not finished just by turning verbs into the passive; the key is to let the reader know why every step exists.

### Counter-example: then/next hides the stopping condition

> The solution was stirred. Then the pH was measured. Next, the sample was collected.

### Good example: Writing the trigger condition

> The solution was stirred until stable pH readings were obtained, at which point the sample was collected.

If waiting for stability affects the result, then is not sufficient information. Run a substitution test on every "subsequently, then, afterwards": can it be written as until, after, before, once or at which point?

### Counter-example: Not stating method modifications

> The assay was performed as previously described.

### Good example: Stating the modifications explicitly

> The assay was performed as previously described, with the following modifications: the incubation time was reduced to 10 min and the buffer was replaced with X.

This is key to reproducibility and attribution. Different methods may produce different results; therefore "according to an existing method" must not mask modifications.

### Good example: Disclosing a small problem and assessing the impact

> Brief contact with the surface occurred during transfer, but the resulting variation was negligible.

This is more credible than "without any complications", because it separates the problem from the impact: the problem exists, the impact is controllable.

Source: Unit 2, 2.2.2, 2.5.1–2.5.3.
