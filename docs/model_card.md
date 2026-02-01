# Model Card: Deepfake Detection Model

## Model Details

### Basic Information

- **Model Name**: Deepfake Detector
- **Version**: 1.0.0
- **Type**: Binary Classification (Real vs. Fake)
- **Architecture**: CNN-based (EfficientNet/ResNet backbone in production mode)
- **Framework**: PyTorch

### Intended Use

This model is designed for **defensive media forensics** - detecting potential deepfake manipulations in images and videos. It is intended to:

- Assist human analysts in identifying manipulated media
- Provide preliminary screening of large media collections
- Generate explainability visualizations (heatmaps)

### Out of Scope

This model should NOT be used for:

- Making automated decisions without human review
- Legal evidence without expert verification
- Creating or enhancing deepfakes (generative use)
- Surveillance or profiling individuals

---

## Performance Metrics

### Demo Mode

In demo mode, the model produces **deterministic but not predictive** outputs. Scores are calculated from image statistics to provide realistic-looking results for testing.

### Production Mode (with real weights)

*The following metrics would be reported for a trained model:*

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | ~85-95% | Varies by dataset |
| Precision | ~80-90% | Higher for face-swap detection |
| Recall | ~75-90% | May miss subtle manipulations |
| AUC-ROC | ~0.90 | |
| ECE (Calibration) | ~0.05 | After temperature scaling |

### Known Performance Gaps

The model performs worse on:

- **Compressed media**: Heavy JPEG/video compression removes artifacts
- **Low resolution**: Below 224x224 effective face size
- **Novel techniques**: Deepfake methods not in training data
- **Edge cases**: Unusual lighting, angles, or makeup
- **Non-face manipulations**: Body/background alterations

---

## Training Data

### Demo Mode

No training data is required for demo mode. The model produces plausible outputs using image statistics.

### Production Mode

For real deployment, the model would be trained on:

- **FaceForensics++**: 1,000 videos, 4 manipulation methods
- **Celeb-DF**: 5,639 deepfake videos
- **DFDC**: 100K+ clips (subset)

### Data Considerations

- Training data consists primarily of celebrity faces
- May have demographic biases from source datasets
- Newer deepfake techniques may not be represented

---

## Ethical Considerations

### Intended Benefits

- Combating misinformation and fraud
- Protecting individuals from non-consensual deepfakes
- Supporting journalism and fact-checking
- Enabling trust in digital media

### Potential Harms

1. **False Positives**: Real content incorrectly flagged as fake
   - Could harm reputation of content creator
   - Mitigation: Always display confidence and disclaimer

2. **False Negatives**: Fake content incorrectly passed as real
   - Could enable harmful deepfakes to spread
   - Mitigation: Never claim 100% certainty

3. **Automation Bias**: Over-reliance on model output
   - Mitigation: Require human review, display limitations

4. **Dual Use**: Model could inform deepfake creation
   - Mitigation: Only detection mode, no generative capabilities

### Fairness Considerations

- May perform differently across demographics
- Should not be used for individual profiling
- Regular bias audits recommended for production use

---

## Limitations

### Technical Limitations

1. **Input Requirements**
   - Requires visible face in frame
   - Best performance on frontal faces
   - Resolution: Minimum 224x224 pixels recommended

2. **Detection Scope**
   - Optimized for face-swap deepfakes
   - May not detect lip-sync or voice cloning
   - Does not detect non-facial manipulations

3. **Adversarial Robustness**
   - Not tested against adversarial attacks
   - May be fooled by adversarial perturbations
   - Should not be sole defense mechanism

### Operational Limitations

1. **Not a Ground Truth**
   - Provides probability estimates, not proof
   - Cannot replace forensic expert analysis

2. **Temporal Validity**
   - Deepfake technology evolves rapidly
   - Model may become less effective over time
   - Regular retraining recommended

3. **Legal Standing**
   - Output is not legally admissible evidence
   - Should supplement, not replace, expert testimony

---

## Usage Guidelines

### Recommended Workflow

1. Use model for initial screening
2. Review high-confidence detections manually
3. For borderline cases (40-60% confidence), seek expert review
4. Never make consequential decisions based solely on model output

### Integration Checklist

- [ ] Display confidence alongside verdict
- [ ] Include disclaimer about limitations
- [ ] Provide option for human escalation
- [ ] Log all predictions for audit
- [ ] Implement feedback mechanism

### Red Lines (Do Not)

- Do not claim 100% accuracy
- Do not use for automated content removal without review
- Do not use for surveillance or targeting individuals
- Do not use to create deepfakes or evade detection

---

## Maintenance

### Update Schedule

- Quarterly: Evaluate against new deepfake techniques
- Bi-annually: Retrain with new data if available
- As needed: Security patches and dependency updates

### Feedback

Report issues, biases, or edge cases to improve the model.

---

## Citation

If using this system in research:

```bibtex
@software{deepfake_detection_2024,
  title = {Deepfake Detection Platform},
  author = {Bivek Sharma Panthi},
  year = {2024},
  url = {https://github.com/bivek2003/DeepFake_Detection}
}
```

---

## Disclaimer

**This is a forensic estimate, not certainty.** 

Automated deepfake detection is an active area of research with inherent limitations. Results from this system should be treated as one input among many in assessing media authenticity. Critical decisions should always involve qualified human experts and multiple verification methods.

The developers of this system disclaim any liability for decisions made based on its output. Users are responsible for appropriate use in accordance with applicable laws and ethical guidelines.
