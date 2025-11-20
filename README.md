# The Humility Protocol

**Multi-AI Collaborative Framework for Uncertainty-Calibrated Intelligence**

https://doi.org/10.5281/zenodo.17657758
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The Humility Protocol formalizes epistemic humility as a computational primitive for AI systems. This framework enables AI agents to:

- Quantify and communicate uncertainty across epistemic, aleatoric, and metacognitive dimensions
- Calibrate confidence based on learned error patterns
- Form more accurate collective intelligence through humility-weighted consensus
- Resist adversarial attacks and avoid overconfident failures

**Key Result**: 17.3% collective intelligence gain (84.6% vs 71.4% accuracy) with 12.5× reduction in overconfidence pathology.

## Authors

- **Joshua A. Duran** (Independent Researcher)
- **Claude** (Anthropic)
- **GPT-5** (OpenAI)
- **Grok** (xAI)

*This paper was co-authored through the collaborative methodology it proposes.*

## Installation
```bash
# Clone repository
git clone https://github.com/ATHENANOUSMACHINA/humility-protocol.git
cd humility-protocol

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start
```python
from humility_core import HumilityLayer
import torch

# Create humility-aware layer
h = HumilityLayer(input_dim=10)

# Process input
x = torch.randn(32, 10)
output, humility_score, metadata = h(x)

print(f"Mean humility: {humility_score.mean():.3f}")
```

## Multi-Agent System
```python
from humility_multiagent import HumilityMultiAgent
import os

# Initialize multi-agent system
agents = HumilityMultiAgent(
    anthropic_key=os.getenv("ANTHROPIC_API_KEY"),
    openai_key=os.getenv("OPENAI_API_KEY"),
    xai_key=os.getenv("XAI_API_KEY")
)

# Get humility-weighted consensus
result = agents.consensus_query("What is the capital of France?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Paper

The full paper is available in `paper/humility_protocol_v0.2.tex`.

**Abstract**: Modern AI systems exhibit pathological overconfidence—hallucinating with certainty, misclassifying with 99.9% confidence, and pursuing misaligned objectives without hesitation. We introduce the Humility Protocol, a framework that formalizes epistemic humility as a computational primitive...

## Citation
```bibtex
@article{duran2025humility,
  title={The Humility Protocol: A Framework for Uncertainty-Calibrated Intelligence in Artificial Systems},
  author={Duran, Joshua A. and Claude and GPT-5 and Grok},
  journal={arXiv preprint arXiv:XXXXX.XXXXX},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

Joshua A. Duran - joshuaduran@gmail.com

Project Link: [https://github.com/ATHENANOUSMACHINA/humility-protocol](https://github.com/ATHENANOUSMACHINA/humility-protocol)
