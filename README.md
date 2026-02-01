
# VGF Model - README.md


# Vijnana Gauge Field Model (VGF)

## Project Overview

This project implements a cognitive computing model based on Buddhist Vijñānavāda philosophy and modern physics gauge field theory. The model simulates the evolution of "Ālaya-vijñāna seeds" in human cognition, exploring the dynamic processes of moral cognition, emotional experience, and conscious activity through multimodal input processing, mental factor analysis, gauge field simulation, and five-hearts state machine tracking.

## Core Features

### 1. Theoretical Integration
- **Buddhist Vijñānavāda Philosophy**: Ālaya-vijñāna, Manas-vijñāna, six root afflictions, five mental states
- **Gauge Field Theory**: Yang-Mills field, structure group mapping, gauge potential evolution
- **Cognitive Science**: Multimodal fusion, short-term memory, conceptual integration

### 2. Multimodal Input System
- **Auditory Modality**: Classical poetry text (e.g., "Guan Ju")
- **Visual Modality**: Emotional color feature vectors
- **Conceptual Modality**: Moral connotation semantic features
- **Dynamic Weight Optimization**: Adjust fusion weights based on modality importance

### 3. Mental Factors Simulation
- **Universal Mental Factors**: Contact, attention, feeling, perception, volition
- **Particular Mental Factors**: Desire, resolution, mindfulness, concentration, wisdom
- **Virtuous Mental Factors**: Non-greed, non-hatred, non-ignorance, etc.
- **Afflictive Mental Factors**: Greed, hatred, ignorance, conceit, doubt, wrong views

### 4. Gauge Field Implementation
- Yang-Mills gauge potential evolution
- Manas four afflictions mapped to SU(2)×U(1) structure group
- Field strength tensor calculation and gauge transformations

### 5. Five Hearts State Machine
- **Initial Response State**: Initial perceptual reaction
- **Seeking State**: Active exploration phase
- **Deciding State**: Cognitive decision-making
- **Purification State**: Moral internalization process
- **Continuous Flow State**: Stable ongoing state

## Project Structure

```

VGF.py
├── Helper Classes (SimpleClassicsProcessor)
├── Parameter Configuration (Params)
├── Core Components
│   ├── Multimodal Input System (MultiModalInput)
│   └── Robust Short-Term Memory (RobustShortTermMemory)
├── Gauge Field Implementation (YangMillsField)
├── Mental Factors System (CompleteMentalFactors)
├── Structure Group Mapping (ManasGaugeMapping)
├── Five Hearts State Machine (ImprovedFiveHeartsStateMachine)
├── Core Cognitive Functions
│   ├── Dynamic Conceptual Integration
│   ├── Feedback Mechanism
│   ├── Imprint Effect
│   └── Ālaya-vijñāna Evolution
├── Visualization System (AdvancedVisualization)
├── Simulation Functions
└── Analysis Functions

```

## Installation Requirements

```bash
pip install numpy matplotlib
```


## Usage

### Basic Execution

```python
python VGF.py
```


### Configuring Export Format

In the `run_comprehensive_simulation()` function, you can set the export format:

- `'pdf'`: Default format, vector graphics
- `'tiff'`: High-resolution bitmap
- `'excel'`: Exports as SVG vector graphics (requires manual insertion into Excel)


### Modifying Input Text

Modify the `SimpleClassicsProcessor.create_guanju_input()` method to change the analyzed text.

## Output Charts

The model generates 9 detailed analysis charts:

1. **Five Hearts State and Multimodal Input**: Shows cognitive state evolution
2. **Gauge Field Dynamics**: Displays gauge potential evolution trajectory
3. **Mental Factors Activity Heatmap**: Intensity changes of five particular mental factors
4. **Seed Evolution Trajectory**: 3D visualization of Ālaya-vijñāna seed evolution
5. **Manas Afflictions Intensity**: Changes in six root afflictions
6. **Consciousness Phase Portrait**: Relationship between consciousness intensity and conceptual integration
7. **State Parallel Coordinates**: Evolution of five-dimensional state vectors
8. **Memory and Integration Strength**: Comparison of memory strength and conceptual integration
9. **Detailed State Transitions Analysis**: State duration, transition frequency, etc.

## Theoretical Significance

1. **Interdisciplinary Integration**: Connects Buddhist philosophy, physics, and cognitive science
2. **Moral Cognition Modeling**: Simulates moral judgment and internalization processes
3. **Consciousness Dynamics**: Captures continuous changes in conscious states
4. **Cultural Analysis**: Deep cognitive interpretation of classical poetry

## Technical Details

### Parameter Configuration

- `STATE_DIM`: 5 (Ālaya-vijñāna seed dimensions)
- `MEMORY_CAPACITY`: 5 (Short-term memory capacity)
- `DECAY_RATE`: 0.9 (Memory decay rate)
- `LEARNING_RATE`: 0.15 (Learning rate)


### Numerical Methods

- Uses NumPy for vector and matrix operations
- Implements Euler method for numerical integration of differential equations
- Includes smoothing filters to prevent numerical oscillations


## Application Scenarios

1. **Cognitive Science Research**: Analysis of consciousness state transitions
2. **Cultural Computing**: Simulation of cognitive experiences with classical literature
3. **AI Ethics**: Modeling of moral decision-making mechanisms
4. **Psychological Research**: Analysis of emotion-cognition interactions

## Future Development Directions

1. Add more modality inputs (tactile, olfactory)
2. Implement long-term memory systems
3. Develop interactive visualization interfaces
4. Extend to analysis of other classical texts
5. Integrate deep learning components

## Important Notes

- Ensure Chinese font support is installed on the system
- Large-scale simulations may require time step adjustments
- Gauge field parameters may need tuning for specific problems


## References

1. *Cheng Weishi Lun* - Xuanzang
2. *Gauge Field Theory* - Yang, Mills
3. *Cognitive Neuroscience* - Gazzaniga
4. *Affective Computing* - Picard

## License

This project is for academic research purposes only.

---

**Project Maintenance**: Buddhist Philosophy and Computational Science Interdisciplinary Research Group
**Version**: 1.0
**Last Updated**: 2024

```

您可以将上述内容保存为 `README.md` 文件，放在您的项目目录中。这个文件包含了完整的英文文档说明，适用于国际化的项目展示。```

