# 02 Introduction: From a Shared Starting Point to a Tractable Gap

## 1. Section Mission and Boundary

The task of the Introduction is not to show how much the author knows, but to let the reader gradually understand: why the research topic is worth attention, what is already known, where it is still insufficient, and why this paper is able to address that insufficiency. It should narrow gradually from a relatively broad shared starting point to the present study; the Discussion normally opens out in the reverse direction.

## 2. The Four-Component Model

```text
I1 Importance of the topic / background facts / general problem
I2 Relevant research and contributions: the research 'map'
I3 gap / problem / motivation / hypothesis / opportunity
I4 The present paper: aim, method overview, main result or contribution
```

The four components are a flexible menu, not a fixed four paragraphs or a fixed sentence count. I1 and I4 appear in almost all articles; I2 may be a short review of current knowledge; I3 may be stated explicitly, or may be implicit in the description of the present study.

## 2.1 Systems-conference variant (governs this project; see guideline 14, SYS-02)

For a CS systems paper, I1–I4 remain the underlying logic but the surface form changes:

- **I1** becomes the workload/technology pressure that makes the problem real *now* (ideally with a number), not general topic importance.
- **I2 shrinks to one or two sentences** naming the closest prior systems and why they do not solve this problem. The full research map moves to a Related Work section near the end of the paper (SYS-09); do not write a literature-review paragraph in a systems Introduction.
- **I3** is the gap plus, crucially, the paper's **key insight** stated in one plain sentence — systems reviewers look for the insight, not just the gap.
- **I4 expands into three mandatory pieces**: (a) the system by name and its 2–3 core mechanisms; (b) an explicit contribution bullet list (3–5 bullets, each verifiable in the Evaluation); (c) a results-preview paragraph with concrete numbers carrying evaluative direction. An Introduction that ends without bullets and numbers reads as a workshop paper.
- Typical length: 1–1.5 columns of a double-column paper; the Introduction must carry the whole case, since reviewers often decide by page 2.

## 3. Rules

### INTRO-01 Use a Shared Starting Point to Let the Reader Enter the Topic

- Rule strength: NORMALLY.
- Writing action: In the first sentence, establish the importance of the topic, current applications, a shared fact or a key definition; when the topic is narrow and the readership is highly specialised, you may go directly into specific facts.
- Sentence/paragraph skeleton: `[Topic] has attracted considerable attention because [reason/value].` / `[Term] is widely used in [context].`
- Tense: `in recent years` is normally followed by the Present Perfect; the current situation takes the Present Simple; if you want long-term validity, prefer an identifiable date rather than a vague recent.
- Why: The reader needs a shared "wall" before they can place the technical "bricks" that follow in the right position. As the book puts it: show them the wall before you start to talk about the bricks.
- Common errors -> Fix action:
  - Starting directly from your own experimental detail -> first give the background the reader needs to understand that detail.
  - Writing only "it is very important" -> give applications, impacts, facts or research trends as the basis.
  - Jumping from too wide to too narrow -> add an intermediate layer, closing the information gaps.
- Completion criterion: The target reader can state the scope and importance of the topic of this paper without first needing to know the author's experiment.
- Source: Unit 1 §1.2.2, §1.4.

### INTRO-02 Organise the Background General-to-Specific

- Rule strength: NORMALLY.
- Writing action: General background -> facts directly related to the problem -> current research focus/problem; a wide readership needs more background than a narrow one.
- Sentence interface: The new concept at the end of one sentence becomes the known information at the start of the next.
- Common errors: Suddenly introducing an unexplained technique, abbreviation or material in the second sentence; mistaking the author's familiarity for the reader's familiarity.
- Completion criterion: The reader can explain how each new term arises from the previous sentence without re-reading.
- Source: Unit 1 §1.2.2, §1.5.2.

### INTRO-03 The Research Map Must Serve the Motivation of This Paper

- Rule strength: MUST.
- Writing action: Select only research directly related to the problem, method or contribution of this paper; organise it according to a general-to-specific, approach/theory/model or chronological pattern.
- Not to do: Arranging the literature you have read by author name into a shopping list; writing the review as a directionless "tennis match" with however bouncing back and forth in every sentence.
- Sentence/paragraph skeleton: `[Author] demonstrated [contribution]. A similar approach was used by [Author], who [next contribution]. To address [issue], [Author]... Taken together, these studies suggest [state leading to gap].`
- Criteria for selecting literature: relevant, essential, and able to show the development of the research and lead the reader towards the problem of this paper.
- Completion criterion: After deleting any one reference, you can explain whether it damages the narrative of "how the research arrived at this paper"; the reader can locate this paper's position on the research map.
- Source: Unit 1 §1.2.2, §1.4.2.

### INTRO-04 State the Gap/Problem Explicitly but with Restraint

- Rule strength: NORMALLY; explicit variant when the research question has already been directly addressed.
- Writing action: First give the known achievements, then point out the specific limitation or unknown, and finally explain why it blocks the aim or requires new research.
- Sentence/paragraph skeleton: `However, [specific limitation] remains unclear.` / `Although [known result], little attention has been paid to [specific gap].` / `[Method] is unable to [needed function].`
- Choose gap type: object, data, mechanism, method, comparison, scope, reproducibility, interpretation or application evidence.
- Language: problem/shortcoming/drawback/lack/unclear/not well understood; a research opportunity may use may/might/could to reduce the assertion risk.
- Not to do: Saying only "little research has been done" without saying which part is little; using aggressive language to negate others; raising the problem suddenly before giving background.
- Completion criterion: The reader can restate "what is currently not known, why it matters, and why this paper is able to address it".
- Source: Unit 1 §1.2.2, §1.4.2.

### INTRO-05 Derive the Aim of This Paper from the Gap, Don't Suddenly Announce the Experiment

- Rule strength: MUST.
- Writing action: Make the aim of this paper respond directly to the preceding gap; use the same key verbs as the gap, avoiding synonym substitution that causes object drift.
- Sentence/paragraph skeleton: `To address this question, we investigated [object] using [approach].` / `The aim of this study was to [verb matching gap].`
- Tense: The content/structure of this paper is normally Present Simple; an already-completed research aim may be Past Simple; when partially achieved or still of current relevance, the Present Simple may be used.
- Subject choice: `This study...`, `The present paper...`, `We...` or the passive are all acceptable, depending on the target journal and whether ownership ambiguity will arise.
- Completion criterion: Place the aim of this paper next to the gap of the previous paragraph, and the two correspond one to one.
- Source: Unit 1 §1.2.2, §1.5.3.

### INTRO-06 Give Only Enough Preview of This Paper's Method and Results

- Rule strength: NORMALLY.
- Writing action: In I4, state what the research does, and if necessary briefly outline the method, main results and achievement; do not move the Methods into the Introduction.
- Sentence/paragraph skeleton: `This paper presents [contribution]. On the basis of [criterion], it describes [approach]. Our results show [key outcome].`
- "Happy words" must serve the evidence: novel, robust, accurate, effective, significantly increased etc. must be supportable by what follows.
- Completion criterion: The reader knows what this paper will report, and can move smoothly into the Methods/Results; they are not swamped by technical detail.
- Source: Unit 1 §1.2.2, §1.4.2.

### INTRO-07 Place Citations in the Right Location

- Rule strength: MUST.
- Writing action: A citation should follow immediately the fact, research action or result it supports; mid-sentence citations are used to demarcate the source of each clause.
- Common errors: Stacking all citations at the end of the sentence, so the reader cannot tell which reference did what; writing a widely accepted fact as if it were a brand-new discovery, or vice versa.
- Tense link: Research findings are initially often Past Simple; once they become accepted facts in the field they may take the Present Simple, and the citation may gradually drop off. Calibrate against recent target articles.
- Completion criterion: Any citation can answer "which part of this sentence it supports, and why it is needed here".
- Source: Unit 1 §1.2.2, §1.4.2; Unit 8 §8.4.

## 4. Paragraph Blueprint

### Paragraph A: Shared Entry and Current Problem

1. Importance or shared fact;
2. Background the reader needs;
3. Narrowing from the background to the general problem or research focus;
4. Closing with the as-yet unresolved limitation.

### Paragraph B: Research Map

1. State how this research direction developed;
2. Group the literature according to one organisational pattern;
3. Compare research methods, contributions and limitations;
4. Use `Taken together...` or an equivalent sentence to drive the map towards the gap.

### Paragraph C: The Present Paper's Response

1. State the gap/problem explicitly;
2. Give the aim, research question or hypothesis;
3. Briefly outline the method of this paper;
4. Preview the main achievement or structure.

## 5. Introduction Completion Checklist

- Is there a shared starting point suited to the target reader?
- Does the background narrow gradually, with no technical jumps?
- Is every citation relevant to the motivation of this paper?
- Is the research map a narrative, not a literature list (systems papers: is it deferred to Related Work, with only the closest work named here)?
- Is the gap specific enough for this paper to address, and is the key insight stated in one sentence?
- Does the goal/hypothesis respond directly to the gap?
- Is the preview of this paper consistent with the actual design and evaluation?
- Is at least one achievement or contribution explicitly identified (systems papers: 3–5 verifiable contribution bullets plus a numeric results preview)?
- Can the last paragraph of the Introduction move naturally into the next section?


## Worked Examples from the Book: How the Gap in the Introduction Is Driven Out

### Counter-example: The first sentence directly announces your own experiment

> In this paper, we investigate the effect of pressure on the mechanical properties of PLA.

### Good example: First establish importance, then narrow the problem

> Biomass-derived PLA has received much attention in recent years because of its potential as a sustainable material.

Then write later:

> Although the effect of A on B has been demonstrated, little attention has been paid to the role of C.

The task of the Introduction is not to say "what I did" as early as possible, but to let the reader gradually accept "why this question is worth doing, how far existing research has gone, and what is still missing".

### Counter-example: The literature is merely juxtaposed

> Smith et al. studied X. Jones et al. studied Y. Lee et al. studied Z.

### Good example: The literature serves the gap

> A pioneering study demonstrated X. Subsequent studies extended this approach to Y. However, whether the same mechanism applies to Z remains unclear.

When checking, tag each reference with a function: starting point, extension, conflict, limitation, method source or gap evidence. If a reference has no function tag, it is probably just literature stacking.

### Counter-example and Good example: Past gap vs current gap

> Little attention was paid to the role of C.

> Little attention has been paid to the role of C.

The former easily confines the gap to the past; the latter connects the gap to the present. If this paper's contribution is precisely to fill a gap that still exists now, you should first check whether the Present Perfect has been mistakenly written as the Past Simple.

### Counter-example: Ending with More research is needed

### Good example: Stating the action of this paper explicitly

> To address this gap, we examined whether C modifies the response of B to A.

Continue with the method or the core finding, so the reader knows how the Methods and Results that follow respond to the Introduction.

Source: Unit 1, especially 1.2.2, 1.5.1–1.5.4.
