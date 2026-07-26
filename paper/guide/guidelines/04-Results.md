# 04 Results: Writing Data as Directional Evidence

## 1. Section Mission and Structure Choice

The task of the Results is not to record item by item the order in which experiments occurred, nor to transcribe the figures and tables, but to arrange the evidence so that the reader can see the trends, comparisons, key results and reasonable implications related to the research aim.

**Systems-paper mapping (governs this project).** In a systems-conference paper this section is called **Evaluation**, and its structure is fixed by guideline 14 SYS-07, not by the journal variants below:

- Open with explicit research questions RQ1…RQn, each tied to a contribution bullet from the Introduction.
- Follow with an **experimental setup** subsection (hardware, workloads, baselines with tuning stated, metrics, statistics protocol).
- Then one subsection per RQ, each **opening with its one-sentence answer**, then the figure/table evidence and its boundaries.
- Required evidence categories: end-to-end vs strongest baselines, per-mechanism ablation, scaling, overheads/costs, sensitivity, and failure behavior where claimed. Every speedup names baseline, endpoint and residency (SYS-08).
- Discussion is usually folded in as a final "Discussion and limitations" subsection.

All evidence-quality rules below (graphic entry, evaluative direction, comparison, honest problems, certainty continuum) apply unchanged to the Evaluation.

The four journal structures (Results + Discussion + Conclusion; Results + Discussion; Results and Discussion + Conclusion; Results + Conclusion) remain relevant only when writing for a journal venue.

## 2. The Four-Component Model

```text
R1 Revisit the aim/hypothesis/literature/method; give an overall result or graphic entry
R2 Key results; evaluation, comparison and direct explanation
R3 Problems, anomalies and limitations in the results; give reasons when needed
R4 Possible implications, applications or interface to the Discussion
```

## 3. Rules

### RES-01 Stabilise the Overall Framework of the Results First

- Rule strength: NORMALLY.
- Writing action: At the start of the Results, revisit the aim, key method, relevant research or overall trend; especially when the Methods are very long, located in supplementary materials, or the reader may be skimming.
- Sentence/paragraph skeleton: `In this study, [method] was used to...` / `Overall, [pattern] was observed.` / `Figure 1 shows [what the reader should inspect].`
- Why: The reader may not have read the Introduction/Methods; re-booting lets all readers understand the results within the same framework.
- Completion criterion: The reader knows which question the following results answer, and which figure or table they need to look at.
- Source: Unit 3 §3.2.1–3.2.2.

### RES-02 Give the Overall Trend Before the Key Details

- Rule strength: NORMALLY.
- Writing action: First give the overall pattern or grouping logic, then select a few key results to write in detail; do not allocate space evenly in the order experiments occurred.
- Why: You know where the research is going, the reader does not; without an overall map, the reader will search for trends themselves and may move towards an interpretation different from yours.
- Completion criterion: The reader knows which result is central and which are merely background or support.
- Source: Unit 3 §3.2.2.

### RES-03 Invite the Reader to Look at Graphics When They Need To

- Rule strength: MUST.
- Writing action: Before a graphic, state its content and reading purpose; after the graphic, state the key observation, comparison or interpretation.
- Sentence/paragraph skeleton: `Figure 1 shows [measure] across [conditions].` / `As can be seen from Figure 1, [interpretive observation].`
- `in Figure` means the content is visible in the figure; `from Figure` means a conclusion can be inferred from the data.
- Graphic order: put the foundational evidence that lets the reader understand the subsequent important results first; it is not necessarily best to put the most exciting figure first.
- Completion criterion: The graphic will not be misread if separated from the surrounding text; the reader knows what to focus on when looking at it.
- Source: Unit 3 §3.2.2, §3.4.2.

### RES-04 Use Evaluative Language to Give Numbers a Direction

- Rule strength: NORMALLY.
- Writing action: After an objective value, add an evaluation related to the research question, such as broadly similar, significantly different, striking reduction, only, as much as, maintained.
- Why: Graphics only show numbers or shapes, not whether the author thinks they are important, strong, weak, typical or anomalous; "23%" may be read as high or low by different readers.
- Note: Evaluative words are not exaggeration; they must match the usage of the target journal, the statistical evidence and the research aim.
- Completion criterion: The reader's understanding of the relative significance and importance of key numbers matches the author's.
- Source: Unit 3 §3.2.2, §3.4.2.

### RES-05 Compare Results to Show the Research Position

- Rule strength: NORMALLY.
- Comparable objects: existing research, predictions, model versus experiment, different conditions, the research hypothesis or different methods.
- Sentence/paragraph skeleton: `In line with [prior work], ...` / `The measured result was consistent with...` / `The model prediction differed from...`
- Why: Original research must state how it adds to, modifies or limits existing knowledge; comparison is the interface where the research map moves from the Introduction into the Results.
- Completion criterion: The object of comparison, the dimension of comparison, and the degree and significance of the similarity/difference are all clear.
- Source: Unit 3 §3.2.2, §3.4.2.

### RES-06 Distinguish Direct Explanation from Full Implication

- Rule strength: MUST.
- A direct explanation may appear in the Results: the material property, method condition or calculation step behind why a result occurred.
- The full implication is usually left for the Discussion, but the end of the Results may open a direction using suggest/indicate/may provide.
- Sentence/paragraph skeleton: `This occurred because [direct reason].` / `These results suggest that [tentative implication].`
- Common errors: Writing suggest as prove; writing a single correlated result as the only causal cause.
- Completion criterion: The reader can separate "what was observed", "directly why", and "what it may imply".
- Source: Unit 3 §3.2.2, §3.5.

### RES-07 Don't Hide Problems, and Turn Them into Manageable Boundaries

- Rule strength: MUST when the problem affects interpretation.
- Writing action: Acknowledge anomalies, uninvestigated factors or incomplete results; minimise the negligible impact, state possible causes, and turn towards future work or positive value.
- Sentence/paragraph skeleton: `The effect of [factor] was not investigated.` / `Although [problem], [impact] was negligible.` / `Nevertheless, the data suggest [supported value].`
- Why: Ignoring a problem makes the author appear not to understand their own research; stating it where it first occurs is more credible than raising it only at the end.
- Completion criterion: The scope of the problem and its actual impact on the main conclusions are both bounded.
- Source: Unit 3 §3.2.2, §3.4.2.

## 4. The Evaluative Language System

### RES-08 Objective Description and Subjective Evaluation Must Be Used Together

- Rule strength: MUST.
- Objective description (is/was + adjective/verb): absent, constant, different, equal, higher/highest, identical, increased, decreased, remained, varied etc. These can be verified as true or false.
- Subjective evaluation: broadly similar, striking reduction, marked increase, significant, dramatic, unexpected etc. These convey the author's judgement of the data.
- **Key distinction**: `higher` is objective (verifiable true/false), but `high` is a subjective assessment. Writing `the value was higher` only reports a fact; writing `the value was high` conveys your judgement of the magnitude.
- Why: If you give only objective numbers without evaluation, different readers will judge for themselves whether the number is high or low, and may move towards an interpretation different from yours, damaging the subsequent conclusions.
- Completion criterion: Every key number or trend has an evaluative direction related to the research question next to it.
- Source: Unit 3 §3.4.2.

### RES-09 Use Evaluative Words to Give Numbers a Direction: Five Groups of Quantity Modifiers

- Rule strength: NORMALLY.
- Writing action: According to the needs of the research, select suitable quantity modifiers from the following five groups:

| Group | Function | Common words |
|---|---|---|
| Increase magnitude | Indicates large/many/high | as many as, at least, considerable, high, large, marked, more than, numerous, significant, substantial |
| Reduce magnitude | Indicates small/few/low | as few as, barely, few, hardly, just, less, little, low, marginal, minimal, minor, modest, only, scarcely, slight, small, under |
| Emphasise degree | Emphasises how big/small | appreciably, considerably, exceptionally, extremely, far, markedly, much, noticeably, particularly, remarkably, significantly, substantially, very, well |
| Indicate closeness | Indicates close to a value | approximately, close to, nearly, negligible, practically, roughly, slightly, virtually |
| Unwilling to direct | Gives no clear direction | fairly, in some cases, moderate, quite, rather, relatively, some, somewhat, to some extent |

- Note: Which group to choose depends on the research aim, the field baseline and the statistical evidence; you must not arbitrarily magnify just to make the results look important.
- Source: Unit 3 §3.4.2.

### RES-10 Frequency Modifiers Have Ten Levels

- Rule strength: NORMALLY.
- Frequency statements are subjective: the same percentage (e.g. 22%) may be described as "frequent" or "rare" depending on prior expectations. If previous research thought a result unlikely, 22% may count as frequent; if previous research thought it very likely, 22% may count as rare.
- The ten levels (from highest to lowest):

```text
1. always, without exception, invariably
2. generally, normally, usually, as a rule
3. frequently, often, commonly, regularly, repeatedly
4. more often than not
5. as often as not (neutral)
6. sometimes, at times, on some occasions
7. occasionally, now and then, from time to time
8. rarely, seldom, infrequently
9. hardly ever, barely ever, almost never
10. never, not once, at no time
```

- Completion criterion: The frequency modifier is consistent with the actually observed frequency and the field expectation.
- Source: Unit 3 §3.4.2.

### RES-11 Comparison Language Must State the Object and the Dimension

- Rule strength: NORMALLY.
- Common comparison phrases: broadly similar to, comparable to, consistent with, contrary to, in good agreement with, in line with, in contrast to, effectively the same as, essentially identical, unlike, well known.
- Common comparison verbs: accord with, align with, compare well with, confirm, contradict, corroborate, deviate from, differ, disprove, mirror, prove, refute, reinforce, resemble, substantiate, support, validate, verify.
- Completion criterion: The object of comparison, the dimension of comparison, and the degree and significance of the similarity/difference are all clear.
- Source: Unit 3 §3.4.2.

### RES-12 The Language of Implication and Explanation Is Graded

- Rule strength: NORMALLY.
- Possibility modifiers: could/may/might be explained by, could be interpreted as, likely, perhaps, possible, presumably, probably, unlikely.
- Inference verbs: deduce, imply, indicate, infer, mean, signify, suggest.
- Inference phrases: it appears that, it could be inferred that, it is evident that, it is probable/likely that, it may be that, it seems that, the evidence suggests that, this implies that, this is indicative of, this seems to suggest that.
- Completion criterion: The inference language matches the position of the evidence on the certainty continuum.
- Source: Unit 3 §3.4.2.

## 5. The Full Sample of the Certainty Continuum

From strongest to most cautious, causality and implication can be progressively de-risked:

```text
1. We found that X causes Y.                                 (direct causality, strongest)
2. We found that X may cause Y.                              (possible causality)
3. We found evidence to suggest that X may be related to Y.  (correlation + possibility)
4. It appears that in some cases, X may have been related to Y.  (bounded scope + possibility)
5. The evidence points to the possibility that in some cases,
   excessive X may have contributed to certain types of Y.   (strongest qualification)
```

- `cause/produce/result in` point to stronger causality; `contribute to` indicates a partial cause; `lead to` indicates a process; `be linked/related/associated with` does not specify a causal direction.
- `results from` and `results in` go in opposite directions.
- The Past Simple usually binds the finding to this study; the Present Simple elevates the finding to a more stable fact, with higher risk and greater force.
- may/might/could, appear to, tend to, in some cases, often etc. must be chosen according to the true position of the evidence; they must not be indiscriminately weakened just to be "safe".
- Risk-reducing language: `it appears that...`, `there is evidence to indicate that...`, as well as frequency qualifiers (often, commonly) and quantity qualifiers (in some cases, in virtually all cases).

### RES-13 Use "!-Substitutes" Instead of Exclamation Marks

- Rule strength: OPTIONAL STRATEGY.
- Science writing does not use exclamation marks, even when the results are exciting. Language that substitutes for the exclamation mark: striking, compelling, crucial, dramatic, excellent, exceptional, exciting, extraordinary, ideal, invaluable, outstanding, perfect, powerful, remarkable, superb, surprising, undeniable, unique, unprecedented, unusual, unquestionably, vital.
- Completion criterion: The importance of a finding is conveyed through evaluative words, not punctuation.
- Source: Unit 3 §3.4.2; Unit 4 §4.4.2.

### RES-14 Present the Best Result as Typical

- Rule strength: OPTIONAL STRATEGY.
- How to write it: first give a general statement, then use `for example` to introduce the best specific result. For example: `The results are generally in very good agreement; for example, at midspan the values are almost identical.`
- Why: This makes the reader feel that the good result is typical, rather than a carefully picked exception.
- Source: Unit 3 §3.2.2.

### RES-15 Re-check the Introduction After Writing the Results

- Rule strength: NORMALLY.
- Writing action: After the Results are complete, go back to the Introduction and check whether the wording of the aim matches the actual results; reword the aim if necessary so that it is consistent with the results.
- Why: During the research, the understanding of the question may deepen, and the original aim may no longer accurately reflect the research output.
- Source: Unit 3 §3.2.2.

## 6. Results Paragraph Blueprint

1. Entry sentence: aim, method, overall trend or subsection question;
2. Evidence sentence: figure, table, measurement, experiment or model result;
3. Evaluation sentence: magnitude, direction, anomaly or reliability;
4. Comparison sentence: literature, prediction, model or control;
5. Explanation sentence: direct mechanism or data source;
6. Limitation sentence: problem, unmeasured factor, anomaly;
7. Interface sentence: cautious implication, pointing towards the Discussion.

## 7. Results Completion Checklist

- Is the order of the results decided by the research question and the reader's needs?
- Is there narrative before and after every main figure or table?
- Do key results receive more explanation and evaluation than secondary results?
- Do comparisons explicitly state "with whom, on which dimension, to what degree"?
- Are results, explanations and speculations separated?
- Do numbers have the necessary directional language such as only/as high as/approximately?
- Are limitations that affect interpretation stated?
- Does the end of the Results open naturally into the Discussion, rather than stopping abruptly?
- Are objective description and subjective evaluation used together? Are there "naked numbers" (numbers with no evaluative direction)?
- Are frequency modifiers consistent with the actually observed frequency and the field expectation?
- After writing the Results, did you re-check the aim in the Introduction?


## Worked Examples from the Book: Fact, Evaluation and Causality in the Results

### Counter-example: Just sending the reader to the figure

> The results are shown in Figure 1.

### Good example: Pointing out the finding in the figure that should be read

> Figure 1 shows a substantial increase in response at high pressure. This increase was observed in all three samples.

The main text must complete at least part of "graphic location - key pattern - consistency or comparison". You cannot assume the reader will automatically know which line, which column or which difference in the figure is most important.

### Counter-example and Good example: Evaluation of the same number

> The response occurred in 23% of cases.

> The response occurred in as many as 23% of cases.

> The response occurred in only 23% of cases.

The evaluative word changes how the number is read. When checking, require the author to state the comparison baseline; do not treat words such as as many as, only, substantial, marked as zero-cost emphasis.

### Counter-example: Writing an observation as proof of mechanism

> These results prove that the treatment changes the response by mechanism X.

### Good example: Preserving the evidence boundary

> These results suggest that the treatment may alter the response.

If the research design did not directly test the mechanism, the Results should separate observation from explanation. caused, proved, demonstrated, suggested, may have resulted from are not freely interchangeable synonyms.

### Causal Strength Check

> X caused Y.

> X contributed to Y.

> X was associated with Y.

> Y may have resulted from X.

From top to bottom, the causal commitment decreases progressively. Which sentence to choose should be decided by the research design and alternative explanations, not by how strong the author wants the results to look.

Source: Unit 3, 3.2.2, 3.3, 3.5.
