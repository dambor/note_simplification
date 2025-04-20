# Medical Notes Simplification with OpenAI API

This project demonstrates how to use Large Language Models (LLMs) through the OpenAI API to simplify medical notes for different patient populations based on their education levels. It evaluates different prompting techniques to determine which ones are most effective for medical text simplification.

## Overview

Medical notes often contain technical terminology that's difficult for patients to understand. This project implements several LLM prompting techniques to transform complex medical notes into simplified explanations tailored to the patient's education level.

## Features

- Simplifies medical notes using 5 different LLM prompting techniques:
  - Basic prompting
  - In-context learning with education-level examples
  - Chain-of-thought reasoning
  - Tree-of-thought reasoning
  - Combined approach (blending in-context learning with tree-of-thought)
  
- Adapts simplifications to different education levels:
  - Elementary
  - High School
  - College
  - Graduate

- Measures and compares the effectiveness of each technique using:
  - Flesch-Kincaid readability scores
  - Average sentence length
  - Medical terminology percentage
  - Visual comparisons and statistical analysis
  
## Project Structure

- `medical_notes_simplification.py`: Main script that processes medical notes with OpenAI API
- `analysis.py`: Visualization and analysis code to evaluate results
- `environment_setup.md`: Instructions for setting up your environment

## Getting Started

1. Follow the instructions in `environment_setup.md` to set up your OpenAI API key and install required packages.
2. Run the main script to process medical notes:
   ```
   python medical_notes_simplification.py
   ```
3. Run the analysis script to visualize the results:
   ```
   python analysis.py
   ```

## Sample Results

The system produces simplified versions of medical notes like this:

**Original:**
> Patient presents with Stage 2 hypertension (BP 162/94). History of myocardial infarction 3 years ago. Currently on amlodipine 5mg daily and atorvastatin 20mg daily. ECG shows left ventricular hypertrophy. Recommend DASH diet and moderate exercise.

**Simplified (High School Level):**
> You have high blood pressure (stage 2). Your blood pressure reading is 162/94, which is too high. You had a heart attack 3 years ago. You're currently taking amlodipine (5mg daily) to lower your blood pressure and atorvastatin (20mg daily) to control your cholesterol. Your heart test (ECG) shows that your heart muscle has become thicker than normal. We recommend you follow a heart-healthy eating plan (DASH diet) and do moderate exercise regularly.

## Conclusion

This project demonstrates how different LLM prompting techniques can be used to make medical information more accessible to patients with varying education levels. The results show that:

1. All prompting methods improve readability compared to original medical notes
2. Tree-of-thought reasoning provides the best structure and organization
3. In-context learning is most effective at adapting to different education levels
4. The combined approach leverages the strengths of both methods

## Future Work

1. Test with a larger and more diverse dataset of medical notes
2. Evaluate with actual patients from different education backgrounds
3. Integrate medical ontologies for better term recognition and replacement
4. Explore multi-modal approaches (adding visual elements to explanations)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
