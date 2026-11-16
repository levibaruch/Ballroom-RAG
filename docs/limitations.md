# Known limitations

**Answers**
- **Answer-model variance.** The same question can give a clearly better or worse answer on another run. Scores from
  one run per question are noisy.
- **Writing-stage problems.** These are the same with every index: repetition across the three teacher parts,
  occasional unrequested extras, and misread step details. The claim judge catches unsupported facts, not weak
  writing.
- **Optional endings get over-generalised.** A chart NOTE that allows an optional ending can turn into a general rule
  in the answer.
- **Latin vs Standard comparisons stay thin.** The books never compare the two styles directly. Only fragments exist.

**Reading the books**
- **Garbled chart cells.** 415 of 1,306 "Timing A/B/C" cells don't split into three parts and stay as printed. The
  other 891 are labelled.
- **Matrix pages.** The colour-matrix reader is verified cell by cell on one page only. It's accepted on 22 pages
  where the result is clean. The other matrix pages stay as text (headers only).

**Search**
- **No auto-read.** An agent that searches a lot and reads little used to get its best search hits read by code.
  That only worked with an older search output format, so it has been inactive since the current format became the
  default (2026-11-14), and the dead code was removed during the cleanup. Bringing it back changes behaviour, so it
  needs an A/B test first.

**Evaluation**
- **The gold set is partly tuning data.** The decision model's label descriptions were tuned on the draft questions,
  so routing scores on them are optimistic. A held-out split is still to do.

**Language and speed**
- **Languages.** English and Dutch are supported. Questions in other languages may be routed out of scope.
- **Speed.** With cloud models a question takes about 8–35 seconds. With a local `gpt-oss:20b` it takes 1–4 minutes.
