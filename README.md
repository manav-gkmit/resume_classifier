# Resume Screening System — Lead Data Scientist Role

## Overview

This project is a deployed resume screening system designed to evaluate candidates for a Lead Data Scientist position.

The system analyzes resumes and generates a selection probability score based on how well a candidate aligns with the target role requirements.

It is built as a decision-support tool for recruiters and hiring teams.

---

## How the System Works

1. **Resume Upload**
   A resume is submitted to the system in PDF format.

2. **Resume Analysis**
   The system analyzes the resume to extract:

   * Professional experience and seniority
   * Leadership and ownership signals
   * Technical depth
   * Evidence of production-grade systems
   * Business impact indicators

3. **Role Alignment Evaluation**
   The resume is compared against the defined Job Description to measure alignment across:

   * Technical skills
   * Project complexity
   * Leadership maturity

4. **Scoring**
   The system generates:

   * A probability score indicating strength of fit
   * A binary decision (Selected / Rejected)

---

## Output

For each resume, the system returns:

* Selection Probability (0 to 1)
* Final Decision (Selected / Rejected)

The probability threshold is configurable depending on hiring criteria.

---