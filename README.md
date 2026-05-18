# Identifying Pipeline Corrosion Risk Using ML for Integrity Management

Published: 2025-10-24
Medium: [https://medium.com/@kyle-t-jones/identifying-pipeline-corrosion-risk-using-ml-for-integrity-management-f185ffd5e007](https://medium.com/@kyle-t-jones/identifying-pipeline-corrosion-risk-using-ml-for-integrity-management-f185ffd5e007)

## Business context

In 2020, Colonial Pipeline suffered a corrosion-related leak in North Carolina that resulted in a loss of containment of ~1.2 million gallons of fuel. Potential issues with the pipeline integrity in that segment had been flagged before but those had been reprioritized (ranked 47th but only the top 30 issues were funded). Colonial had to$7.8 million in cleanup costs and penalties.

Traditional corrosion risk models use linear scoring: add points for age, subtract points for cathodic protection (CP) readings, multiply by consequence factors. But corrosion is nonlinear. The interaction between soil resistivity and CP potential isn't additive --- it's multiplicative and threshold-driven. A 40-year-old pipeline with marginal CP in low-resistivity soil behaves fundamentally differently than the sum of its parts would suggest.

Machine learning doesn't just score risk --- it learns these interactions from data. A gradient boosting classifier trained on inline inspection (ILI) results, CP surveys, soil conditions, and coating assessments produces risk rankings that optimize inspection budgets by focusing resources where failure probability intersects with consequence.

## About

Place the code for this article in this repository.
The original article export is saved as `article.md`.

## Files

Add your `.ipynb`, `.py`, `.yaml`, `.js`, `.ts`, or other project files here.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).