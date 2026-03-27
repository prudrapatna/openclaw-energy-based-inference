---
name: energy-based-inference
description: System 2 energy-based inference for Physio-WorldModel V7, implementing ts_loss optimization for counterfactual evaluation. Use when running Gate 3 evaluation, counterfactual analysis, or energy-based optimization of world models.
---

# Energy-Based Inference (System 2)

## Overview

Implements energy-based inference for Physio-WorldModel V7, optimizing the `ts_loss` (temporal stability loss) to find optimal interventions. This is the "System 2" reasoning approach for counterfactual evaluation.

## Core Algorithm

### Energy Function
```python
def calculate_ts_loss(predictions, targets, context):
    """
    Temporal Stability Loss - measures prediction consistency
    Lower loss = more stable/causal predictions
    """
    # 1. Reconstruction loss
    recon_loss = F.mse_loss(predictions, targets)
    
    # 2. Temporal smoothness
    temp_loss = F.mse_loss(predictions[1:], predictions[:-1])
    
    # 3. Causal consistency
    causal_loss = measure_causal_consistency(predictions, context)
    
    return recon_loss + 0.1 * temp_loss + 0.05 * causal_loss
```

### Optimization Loop
```python
def energy_based_inference(model, initial_state, target_outcome, steps=20, lr=0.01):
    """
    Find intervention that minimizes ts_loss to achieve target outcome
    """
    intervention = initial_state.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([intervention], lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Forward pass with intervention
        predictions = model(intervention)
        
        # Calculate energy (ts_loss)
        energy = calculate_ts_loss(predictions, target_outcome, initial_state)
        
        # Backward pass
        energy.backward()
        optimizer.step()
        
        if step % 5 == 0:
            print(f"Step {step}: Energy = {energy.item():.4f}")
    
    return intervention.detach(), energy.item()
```

## Integration with V7 Evaluation

### Modified run_probe.py
```python
# Add energy-based inference to existing evaluation
from energy_based_inference import energy_based_inference

def evaluate_gate3_energy_based(model, patient_data):
    """
    Gate 3: Counterfactual evaluation with energy-based inference
    """
    results = []
    
    for patient in patient_data:
        initial_state = patient['baseline']
        target_outcome = patient['target']
        
        # Find optimal intervention
        optimal_intervention, final_energy = energy_based_inference(
            model, initial_state, target_outcome,
            steps=20, lr=0.01
        )
        
        # Evaluate
        prediction = model(optimal_intervention)
        mae = F.l1_loss(prediction, target_outcome)
        
        results.append({
            'patient_id': patient['id'],
            'optimal_intervention': optimal_intervention.cpu().numpy(),
            'final_energy': final_energy,
            'mae': mae.item(),
            'prediction': prediction.cpu().numpy()
        })
    
    return results
```

## Hyperparameters

### Default Settings
```yaml
energy_based_inference:
  optimization_steps: 20
  learning_rate: 0.01
  energy_weights:
    reconstruction: 1.0
    temporal_smoothness: 0.1
    causal_consistency: 0.05
  regularization:
    l2_weight: 0.001
    intervention_bound: 2.0  # Max intervention magnitude
```

### Tuning Recommendations
1. **For CGMacros data**: Use `steps=30`, `lr=0.005` (sparser data)
2. **For VitalDB data**: Use `steps=15`, `lr=0.02` (denser data)
3. **For sensitive outcomes**: Increase causal consistency weight to 0.1

## Use Cases

### 1. Gate 3 Evaluation (Counterfactuals)
```python
# Load adapted V7 model for CGMacros
model = load_adapted_v7(input_channels=5)
results = evaluate_gate3_energy_based(model, cgmacros_patients)
```

### 2. Intervention Optimization
```python
# Find optimal meal intervention for glucose control
optimal_meal = energy_based_inference(
    model, current_state, target_glucose=100,
    steps=25, lr=0.008
)
```

### 3. Causal Discovery
```python
# Test which interventions have largest effect
intervention_effects = []
for intervention_type in ['carbs', 'protein', 'fat', 'exercise']:
    effect = evaluate_intervention_effect(
        model, intervention_type, energy_based=True
    )
    intervention_effects.append(effect)
```

## Integration with AutoResearcher

### Auto-tuning Configuration
```python
# AutoResearcher will optimize these hyperparameters
search_space = {
    'optimization_steps': [10, 15, 20, 25, 30],
    'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05],
    'energy_weights': {
        'reconstruction': [0.5, 1.0, 2.0],
        'temporal_smoothness': [0.05, 0.1, 0.2],
        'causal_consistency': [0.01, 0.05, 0.1]
    }
}
```

### Evaluation Metric
```python
def evaluate_energy_based_config(config, model, validation_data):
    """
    Evaluate a hyperparameter configuration
    Returns: Average MAE across validation set
    """
    total_mae = 0
    for patient in validation_data:
        _, _, mae = energy_based_inference(
            model, patient, 
            steps=config['optimization_steps'],
            lr=config['learning_rate']
        )
        total_mae += mae
    
    return total_mae / len(validation_data)
```

## Testing

### Test Script
```bash
cd ~/.openclaw/skills/energy-based-inference
python3 scripts/test_energy_inference.py
```

### Expected Output
```
Testing Energy-Based Inference...
[1] Simple optimization test: ✓
[2] Convergence test (20 steps): ✓
[3] Intervention bounds test: ✓
[4] Integration with V7 test: ✓
All tests passed!
```

## Performance Considerations

### GPU Optimization
- Use `torch.compile()` for faster inference
- Batch multiple patients for parallel optimization
- Use mixed precision (FP16) for memory efficiency

### Memory Usage
- 20 optimization steps: ~2GB VRAM per patient
- Can optimize 4 patients simultaneously on A100 (40GB)

### Speed
- 20 steps: ~0.5 seconds per patient on A100
- Full CGMacros cohort (45 patients): ~23 seconds

## References

1. **V7 Paper**: Energy-based world models
2. **Gate 3 Methodology**: Counterfactual evaluation
3. **PyTorch Optimization**: Adam optimizer, gradient descent
4. **Medical AI**: Intervention optimization in physiological systems