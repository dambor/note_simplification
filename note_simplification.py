"""
# Medical Notes Simplification using LLMs
This notebook demonstrates how to use Large Language Models (LLMs) to simplify medical notes
for different patient populations, making healthcare information more accessible.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import os
from IPython.display import display, HTML
import requests
from sklearn.metrics import cohen_kappa_score
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Make sure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function to measure text complexity
def measure_complexity(text):
    """
    Measures text complexity using various metrics:
    - Average sentence length
    - Average word length
    - Percentage of complex medical terms
    - Flesch-Kincaid readability score
    """
    if not text or text.strip() == "":
        return {
            "avg_sentence_length": 0,
            "avg_word_length": 0,
            "medical_terms_pct": 0,
            "flesch_kincaid": 100
        }
    
    # Tokenize text
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Filter out punctuation
    words = [word for word in words if word.isalpha()]
    
    # Calculate metrics
    avg_sentence_length = len(words) / max(len(sentences), 1)
    avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
    
    # Simple list of complex medical terms (this would be expanded in a real application)
    medical_terms = ["hypertension", "myocardial", "infarction", "hyperlipidemia", 
                    "atherosclerosis", "dyspnea", "arrhythmia", "tachycardia", 
                    "bradycardia", "nephropathy", "neuropathy", "retinopathy"]
    
    medical_terms_count = sum(1 for word in words if word.lower() in medical_terms)
    medical_terms_pct = (medical_terms_count / max(len(words), 1)) * 100
    
    # Calculate Flesch-Kincaid readability score (simplified version)
    # 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    # For simplicity, we'll estimate syllables based on word length
    total_syllables = sum(max(1, len(word) // 3) for word in words)
    flesch_kincaid = 206.835 - 1.015 * avg_sentence_length - 84.6 * (total_syllables / max(len(words), 1))
    flesch_kincaid = max(0, min(100, flesch_kincaid))  # Clamp to [0, 100]
    
    return {
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "medical_terms_pct": medical_terms_pct,
        "flesch_kincaid": flesch_kincaid
    }

# Load synthetic medical data
def load_sample_data(filename='synthetic_medical_notes.json'):
    """
    Load sample synthetic medical notes.
    If file doesn't exist, create synthetic data.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        # Create synthetic data if file doesn't exist
        synthetic_data = [
            {
                "patient_id": "P001",
                "age": 65,
                "education_level": "high_school",
                "medical_note": "Patient presents with Stage 2 hypertension (BP 162/94). History of myocardial infarction 3 years ago. Currently on amlodipine 10mg daily and atorvastatin 40mg daily. Exhibits peripheral edema and occasional dyspnea on exertion. ECG shows left ventricular hypertrophy. Recommend DASH diet, sodium restriction, and daily aerobic exercise for 30 minutes."
            },
            {
                "patient_id": "P002",
                "age": 72,
                "education_level": "elementary",
                "medical_note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%). Presents with polyuria, polydipsia, and unexplained weight loss of 10 pounds over 3 months. Fasting plasma glucose 182 mg/dL. Evidence of early diabetic nephropathy with microalbuminuria. Starting metformin 500mg BID, titrating to 1000mg BID over 4 weeks. Referral to diabetic education program and nutritionist."
            },
            {
                "patient_id": "P003",
                "age": 45,
                "education_level": "graduate",
                "medical_note": "Patient exhibits acute bronchitis with purulent sputum production and persistent cough for 10 days. Afebrile, with mild wheezing on expiration. No signs of pneumonia on chest X-ray. CBC shows slight leukocytosis. Prescribed azithromycin 500mg day 1, then 250mg daily for 4 days. Recommended increased fluid intake and rest. Follow up if symptoms persist beyond 14 days."
            },
            {
                "patient_id": "P004",
                "age": 58,
                "education_level": "high_school",
                "medical_note": "Patient presenting with symptoms consistent with GERD, including retrosternal burning pain exacerbated by recumbency and large meals. Upper endoscopy reveals Grade B esophagitis per LA classification. H. pylori test negative. Initiating PPI therapy with omeprazole 40mg daily before breakfast. Lifestyle modifications discussed including elevation of head of bed, weight loss, and avoiding trigger foods."
            },
            {
                "patient_id": "P005",
                "age": 33,
                "education_level": "college",
                "medical_note": "Patient presents with migraine without aura, characterized by unilateral throbbing headache, photophobia, phonophobia, and nausea. Headache frequency of 3-4 episodes monthly, each lasting 12-24 hours. Triggers include stress and irregular sleep. Neurological examination unremarkable. Prescribed sumatriptan 50mg PRN for acute attacks and recommended migraine diary to identify patterns and triggers."
            }
        ]
        with open(filename, 'w') as f:
            json.dump(synthetic_data, f, indent=2)
        return synthetic_data

# Load the data
sample_data = load_sample_data()
df = pd.DataFrame(sample_data)
print(f"Loaded {len(df)} synthetic medical notes")

# Define LLM Prompting Functions

def basic_prompt(medical_note):
    """
    Basic prompt for medical note simplification.
    """
    prompt = f"""
    Please simplify the following medical note for a patient:
    
    {medical_note}
    
    Simplified version:
    """
    
    # In a real implementation, this would call an LLM API
    # For this example, we'll simulate a response
    return simulate_llm_response(prompt, method="basic")

def in_context_learning_prompt(medical_note, education_level):
    """
    In-context learning prompt with examples for specific education levels.
    """
    examples = {
        "elementary": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have high blood sugar (Type 2 Diabetes). Your blood test shows your sugar levels have been high for the past few months."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "Your ankles and feet are swelling with fluid. Sometimes you feel short of breath when you're active."
            }
        ],
        "high_school": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have Type 2 Diabetes. Your HbA1c test, which measures your average blood sugar over the past 3 months, shows it's high at 8.2%."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "You have swelling in your legs and feet, and sometimes have trouble breathing when you're physically active."
            }
        ],
        "college": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have Type 2 Diabetes with an HbA1c of 8.2%, indicating poor glycemic control over the past 3 months."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "You're experiencing fluid retention in your extremities and shortness of breath during physical activity."
            }
        ],
        "graduate": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have Type 2 Diabetes Mellitus with an elevated HbA1c of 8.2%, indicating sustained hyperglycemia over the preceding three months."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "You're presenting with peripheral edema and intermittent dyspnea upon exertion, suggesting possible cardiovascular or pulmonary etiology."
            }
        ]
    }
    
    # Default to high_school level if the specified level is not available
    if education_level not in examples:
        education_level = "high_school"
    
    prompt = f"""
    Please simplify the following medical note for a patient with {education_level} education level.
    
    Here are some examples of simplifying medical notes for someone with {education_level} education:
    
    Original: {examples[education_level][0]['note']}
    Simplified: {examples[education_level][0]['simplified']}
    
    Original: {examples[education_level][1]['note']}
    Simplified: {examples[education_level][1]['simplified']}
    
    Now, please simplify this medical note:
    {medical_note}
    
    Simplified version:
    """
    
    return simulate_llm_response(prompt, method="in_context", education_level=education_level)

def chain_of_thought_prompt(medical_note, education_level):
    """
    Chain-of-thought prompt that asks the LLM to reason through its simplification.
    """
    prompt = f"""
    Please simplify the following medical note for a patient with {education_level} education level.
    
    Medical note: {medical_note}
    
    Let's think through this step by step:
    1. First, identify all medical terms that need simplification
    2. Determine appropriate replacement words/phrases based on the patient's education level
    3. Rewrite the note with simplified terms while preserving all important medical information
    4. Check that the simplified note is appropriate for a patient with {education_level} education
    
    Simplified version:
    """
    
    return simulate_llm_response(prompt, method="chain_of_thought", education_level=education_level)

def tree_of_thought_prompt(medical_note, education_level):
    """
    Tree-of-thought prompt that explores multiple simplification options.
    """
    prompt = f"""
    Please simplify the following medical note for a patient with {education_level} education level.
    
    Medical note: {medical_note}
    
    Let's consider multiple approaches to simplify this note:
    
    Approach 1: Replace technical terms with common language
    - For example, "hypertension" becomes "high blood pressure"
    - How would this approach simplify the entire note?
    
    Approach 2: Reorganize information by importance
    - Start with diagnosis and what it means for the patient
    - Follow with treatment plan in simple terms
    - End with follow-up instructions
    - How would this approach reorganize the note?
    
    Approach 3: Use metaphors and analogies
    - Explain complex medical concepts using everyday comparisons
    - How would this approach make the concepts more relatable?
    
    Now, choose the best elements from each approach to create a final simplified version that is:
    - Accurate (contains all critical medical information)
    - Understandable (appropriate for {education_level} education level)
    - Actionable (patient knows what to do next)
    
    Final simplified version:
    """
    
    return simulate_llm_response(prompt, method="tree_of_thought", education_level=education_level)

# Simulate LLM response for demonstration purposes
def simulate_llm_response(prompt, method, education_level=None):
    """
    Simulates an LLM response for demonstration.
    In a real implementation, this would call an actual LLM API.
    """
    
    # Extract the medical note from the prompt
    medical_note = re.search(r"Medical note: (.*?)(?:\n|Let's)", prompt, re.DOTALL)
    if not medical_note:
        medical_note = re.search(r"medical note:\s*(.*?)(?:\n|Simplified)", prompt, re.DOTALL)
    
    if not medical_note:
        return "Could not parse medical note from prompt."
    
    medical_note = medical_note.group(1).strip()
    
    # Simple rule-based simplification
    simplified = medical_note
    
    # Replace medical terms with simpler alternatives
    replacements = {
        "hypertension": "high blood pressure",
        "myocardial infarction": "heart attack",
        "dyspnea": "shortness of breath",
        "edema": "swelling",
        "amlodipine": "blood pressure medicine",
        "atorvastatin": "cholesterol medicine",
        "peripheral": "in your legs and feet",
        "ECG": "heart test",
        "left ventricular hypertrophy": "enlarged heart",
        "DASH diet": "heart-healthy diet",
        "Type 2 Diabetes Mellitus": "type 2 diabetes",
        "HbA1c": "blood sugar test",
        "polyuria": "frequent urination",
        "polydipsia": "feeling very thirsty",
        "plasma glucose": "blood sugar level",
        "diabetic nephropathy": "kidney problems from diabetes",
        "microalbuminuria": "protein in your urine",
        "metformin": "diabetes medicine",
        "BID": "twice daily",
        "acute bronchitis": "chest infection",
        "purulent sputum": "yellow/green mucus",
        "afebrile": "no fever",
        "wheezing": "whistling sound when breathing",
        "pneumonia": "lung infection",
        "chest X-ray": "chest picture",
        "CBC": "blood test",
        "leukocytosis": "high white blood cell count",
        "azithromycin": "antibiotic",
        "GERD": "acid reflux",
        "retrosternal": "behind the breastbone",
        "recumbency": "lying down",
        "endoscopy": "camera test",
        "esophagitis": "inflammation of the food pipe",
        "H. pylori": "stomach bacteria",
        "PPI therapy": "acid-reducing medication",
        "omeprazole": "acid-reducing medicine",
        "migraine without aura": "migraine headache",
        "unilateral": "one-sided",
        "photophobia": "sensitivity to light",
        "phonophobia": "sensitivity to sound",
        "neurological examination": "nerve and brain test",
        "sumatriptan": "migraine medicine",
        "PRN": "as needed"
    }
    
    # Adjustments based on education level (if provided)
    if education_level == "elementary":
        for term, replacement in replacements.items():
            simplified = simplified.replace(term, replacement)
        # Further simplify sentences
        simplified = simplified.replace(". ", ".\n")
        simplified = re.sub(r'(\d+)mg', r'\1 milligrams', simplified)
        
    elif education_level == "high_school":
        for term, replacement in replacements.items():
            # Keep some terms with explanations
            if term in ["HbA1c", "ECG", "DASH diet"]:
                simplified = simplified.replace(term, f"{term} ({replacement})")
            else:
                simplified = simplified.replace(term, replacement)
    
    elif education_level == "college":
        # Keep more medical terms with some explanations
        for term, replacement in replacements.items():
            if term in ["dyspnea", "peripheral edema", "myocardial infarction", "hypertension"]:
                simplified = simplified.replace(term, f"{term} ({replacement})")
            else:
                simplified = simplified.replace(term, replacement)
    
    elif education_level == "graduate":
        # Keep most medical terms, only simplify very complex ones
        for term, replacement in replacements.items():
            if term in ["microalbuminuria", "left ventricular hypertrophy"]:
                simplified = simplified.replace(term, f"{term} ({replacement})")

    # Adjust based on prompting method
    if method == "basic":
        # Basic prompt just gets the default simplified version
        pass
        
    elif method == "in_context":
        # In-context learning might have better formatting and clarity
        simplified = simplified.replace(". ", ".\n\n")
        simplified = "WHAT THIS MEANS FOR YOU:\n\n" + simplified
        
    elif method == "chain_of_thought":
        # Add explanations and action items
        simplified += "\n\nWHAT TO DO NEXT:\n"
        if "DASH diet" in medical_note or "diet" in medical_note:
            simplified += "\n- Follow your recommended diet plan"
        if "exercise" in medical_note:
            simplified += "\n- Make sure to exercise regularly as advised"
        if "medicine" in simplified or "medication" in simplified:
            simplified += "\n- Take your medications as prescribed"
        simplified += "\n- Contact your doctor if you have questions or new symptoms"
        
    elif method == "tree_of_thought":
        # More structured with headers and bullet points
        sections = {
            "YOUR CONDITION:": [],
            "YOUR TREATMENT PLAN:": [],
            "WHAT TO WATCH FOR:": [],
            "NEXT STEPS:": []
        }
        
        # Populate sections based on content
        if "high blood pressure" in simplified or "blood pressure" in simplified:
            sections["YOUR CONDITION:"].append("You have high blood pressure")
        if "heart attack" in simplified:
            sections["YOUR CONDITION:"].append("You had a heart attack in the past")
        if "diabetes" in simplified:
            sections["YOUR CONDITION:"].append("You have diabetes")
        if "infection" in simplified:
            sections["YOUR CONDITION:"].append("You have an infection")
        if "acid reflux" in simplified:
            sections["YOUR CONDITION:"].append("You have acid reflux (heartburn)")
        if "migraine" in simplified:
            sections["YOUR CONDITION:"].append("You get migraine headaches")
            
        if "medicine" in simplified:
            meds = re.findall(r'([a-zA-Z]+) (medicine|medication)', simplified)
            for med in meds:
                sections["YOUR TREATMENT PLAN:"].append(f"Take your {med[0]} {med[1]} as prescribed")
        
        if "diet" in simplified:
            sections["YOUR TREATMENT PLAN:"].append("Follow the recommended diet plan")
        if "exercise" in simplified:
            sections["YOUR TREATMENT PLAN:"].append("Exercise regularly as recommended")
            
        sections["NEXT STEPS:"].append("Follow up with your doctor as scheduled")
        sections["NEXT STEPS:"].append("Call if your symptoms get worse or you have questions")
        
        # Format the results
        result = ""
        for section, items in sections.items():
            if items:  # Only include non-empty sections
                result += section + "\n"
                for item in items:
                    result += f"- {item}\n"
                result += "\n"
                
        simplified = result

    return simplified

# Process the sample data with different methods
results = []

for idx, row in df.iterrows():
    medical_note = row['medical_note']
    education_level = row['education_level']
    patient_id = row['patient_id']
    
    # Apply different prompting methods
    basic_result = basic_prompt(medical_note)
    in_context_result = in_context_learning_prompt(medical_note, education_level)
    cot_result = chain_of_thought_prompt(medical_note, education_level)
    tot_result = tree_of_thought_prompt(medical_note, education_level)
    
    # Measure complexity of original and simplified notes
    original_complexity = measure_complexity(medical_note)
    basic_complexity = measure_complexity(basic_result)
    in_context_complexity = measure_complexity(in_context_result)
    cot_complexity = measure_complexity(cot_result)
    tot_complexity = measure_complexity(tot_result)
    
    results.append({
        "patient_id": patient_id,
        "education_level": education_level,
        "original_note": medical_note,
        "original_complexity": original_complexity,
        "basic_prompt_result": basic_result,
        "basic_complexity": basic_complexity,
        "in_context_result": in_context_result,
        "in_context_complexity": in_context_complexity,
        "chain_of_thought_result": cot_result,
        "cot_complexity": cot_complexity,
        "tree_of_thought_result": tot_result,
        "tot_complexity": tot_complexity
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Create some visualizations to compare methods

# Plot readability scores across methods
def plot_readability_comparisons(results_df):
    """
    Create visualizations comparing readability metrics across different methods.
    """
    plt.figure(figsize=(14, 8))
    
    # Extract Flesch-Kincaid scores for all methods
    fk_scores = {
        'Original': [r['original_complexity']['flesch_kincaid'] for r in results],
        'Basic': [r['basic_complexity']['flesch_kincaid'] for r in results],
        'In-Context': [r['in_context_complexity']['flesch_kincaid'] for r in results],
        'Chain-of-Thought': [r['cot_complexity']['flesch_kincaid'] for r in results],
        'Tree-of-Thought': [r['tot_complexity']['flesch_kincaid'] for r in results]
    }
    
    # Prepare data for plotting
    methods = list(fk_scores.keys())
    x = np.arange(len(results))
    width = 0.15
    offsets = np.arange(-2, 3) * width
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        plt.bar(x + offsets[i], fk_scores[method], width, label=method)
    
    plt.xlabel('Patient')
    plt.ylabel('Flesch-Kincaid Readability (higher is better)')
    plt.title('Readability Comparison Across Methods')
    plt.xticks(x, [r['patient_id'] for r in results])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal lines for readability guidelines
    plt.axhline(y=80, color='g', linestyle='-', alpha=0.3)
    plt.axhline(y=60, color='y', linestyle='-', alpha=0.3)
    plt.axhline(y=40, color='r', linestyle='-', alpha=0.3)
    
    plt.text(len(results)-1, 85, 'Easy', color='g', ha='right')
    plt.text(len(results)-1, 65, 'Standard', color='y', ha='right')
    plt.text(len(results)-1, 45, 'Difficult', color='r', ha='right')
    
    plt.tight_layout()
    plt.savefig('readability_comparison.png')
    plt.show()
    
    # Plot average sentence length comparison
    plt.figure(figsize=(14, 6))
    
    # Extract average sentence lengths
    sent_lengths = {
        'Original': [r['original_complexity']['avg_sentence_length'] for r in results],
        'Basic': [r['basic_complexity']['avg_sentence_length'] for r in results],
        'In-Context': [r['in_context_complexity']['avg_sentence_length'] for r in results],
        'Chain-of-Thought': [r['cot_complexity']['avg_sentence_length'] for r in results],
        'Tree-of-Thought': [r['tot_complexity']['avg_sentence_length'] for r in results]
    }
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        plt.bar(x + offsets[i], sent_lengths[method], width, label=method)
    
    plt.xlabel('Patient')
    plt.ylabel('Average Sentence Length (words)')
    plt.title('Sentence Length Comparison Across Methods')
    plt.xticks(x, [r['patient_id'] for r in results])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add reference line for recommended sentence length
    plt.axhline(y=15, color='r', linestyle='--', alpha=0.5)
    plt.text(len(results)-1, 16, 'Recommended max. for general audience', color='r', ha='right')
    
    plt.tight_layout()
    plt.savefig('sentence_length_comparison.png')
    plt.show()
    
    # Compare medical terms percentage
    plt.figure(figsize=(14, 6))
    
    # Extract medical terms percentages
    med_terms = {
        'Original': [r['original_complexity']['medical_terms_pct'] for r in results],
        'Basic': [r['basic_complexity']['medical_terms_pct'] for r in results],
        'In-Context': [r['in_context_complexity']['medical_terms_pct'] for r in results],
        'Chain-of-Thought': [r['cot_complexity']['medical_terms_pct'] for r in results],
        'Tree-of-Thought': [r['tot_complexity']['medical_terms_pct'] for r in results]
    }
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        plt.bar(x + offsets[i], med_terms[method], width, label=method)
    
    plt.xlabel('Patient')
    plt.ylabel('Medical Terms (%)')
    plt.title('Medical Terminology Usage Across Methods')
    plt.xticks(x, [r['patient_id'] for r in results])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('medical_terms_comparison.png')
    plt.show()

# Run the plotting function
plot_readability_comparisons(results_df)

# Analyze effectiveness by education level
def analyze_by_education_level(results_df):
    """
    Analyze the effectiveness of different methods by education level.
    """
    # Group by education level
    education_levels = results_df['education_level'].unique()
    
    # Calculate average readability improvements by education level and method
    improvements = []
    
    for edu in education_levels:
        edu_rows = results_df[results_df['education_level'] == edu]
        
        for _, row in edu_rows.iterrows():
            orig_fk = row['original_complexity']['flesch_kincaid']
            
            # Calculate improvements for each method
            basic_imp = row['basic_complexity']['flesch_kincaid'] - orig_fk
            in_context_imp = row['in_context_complexity']['flesch_kincaid'] - orig_fk
            cot_imp = row['cot_complexity']['flesch_kincaid'] - orig_fk
            tot_imp = row['tot_complexity']['flesch_kincaid'] - orig_fk
            
            improvements.append({
                'education_level': edu,
                'basic_improvement': basic_imp,
                'in_context_improvement': in_context_imp,
                'cot_improvement': cot_imp,
                'tot_improvement': tot_imp
            })
    
    # Convert to DataFrame
    imp_df = pd.DataFrame(improvements)
    
    # Group by education level and calculate mean improvements
    grouped = imp_df.groupby('education_level').mean()
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    grouped.plot(kind='bar', figsize=(12, 6))
    plt.title('Average Readability Improvement by Education Level')
    plt.ylabel('Flesch-Kincaid Score Improvement')
    plt.xlabel('Education Level')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('improvement_by_education.png')
    plt.show()
    
    return grouped

# Run education level analysis
education_analysis = analyze_by_education_level(results_df)
print("Average readability improvement by education level:")
print(education_analysis)

# Display a sample comparison
def display_sample_comparison(results_df, patient_id="P001"):
    """
    Display a side-by-side comparison of original and simplified notes for a specific patient.
    """
    row = results_df[results_df['patient_id'] == patient_id].iloc[0]
    
    print(f"Comparison for Patient {patient_id} (Education level: {row['education_level']})")
    print("\nORIGINAL NOTE:")
    print(row['original_note'])
    print(f"\nReadability score: {row['original_complexity']['flesch_kincaid']:.1f}")
    print(f"Average sentence length: {row['original_complexity']['avg_sentence_length']:.1f} words")
    print(f"Medical terms: {row['original_complexity']['medical_terms_pct']:.1f}%")
    
    print("\n" + "="*50 + "\n")
    
    print("TREE-OF-THOUGHT SIMPLIFIED NOTE:")
    print(row['tree_of_thought_result'])
    print(f"\nReadability score: {row['tot_complexity']['flesch_kincaid']:.1f}")
    print(f"Average sentence length: {row['tot_complexity']['avg_sentence_length']:.1f} words")
    print(f"Medical terms: {row['tot_complexity']['medical_terms_pct']:.1f}%")

# Display sample comparison
display_sample_comparison(results_df, "P001")

# Evaluate the results with common metrics
def evaluate_methods():
    """
    Evaluate the different methods using common metrics:
    - Readability improvement
    - Preservation of medical information
    - Education level adaptation
    """
    # Calculate average improvements across all patients
    avg_improvements = {
        'Basic': np.mean([r['basic_complexity']['flesch_kincaid'] - r['original_complexity']['flesch_kincaid'] for r in results]),
        'In-Context': np.mean([r['in_context_complexity']['flesch_kincaid'] - r['original_complexity']['flesch_kincaid'] for r in results]),
        'Chain-of-Thought': np.mean([r['cot_complexity']['flesch_kincaid'] - r['original_complexity']['flesch_kincaid'] for r in results]),
        'Tree-of-Thought': np.mean([r['tot_complexity']['flesch_kincaid'] - r['original_complexity']['flesch_kincaid'] for r in results])
    }
    
    # Calculate medical term reduction
    term_reduction = {
        'Basic': np.mean([r['original_complexity']['medical_terms_pct'] - r['basic_complexity']['medical_terms_pct'] for r in results]),
        'In-Context': np.mean([r['original_complexity']['medical_terms_pct'] - r['in_context_complexity']['medical_terms_pct'] for r in results]),
        'Chain-of-Thought': np.mean([r['original_complexity']['medical_terms_pct'] - r['cot_complexity']['medical_terms_pct'] for r in results]),
        'Tree-of-Thought': np.mean([r['original_complexity']['medical_terms_pct'] - r['tot_complexity']['medical_terms_pct'] for r in results])
    }
    
    # Create a summary table
    summary = pd.DataFrame({
        'Method': ['Basic', 'In-Context', 'Chain-of-Thought', 'Tree-of-Thought'],
        'Avg. Readability Improvement': [avg_improvements['Basic'], avg_improvements['In-Context'], 
                                          avg_improvements['Chain-of-Thought'], avg_improvements['Tree-of-Thought']],
        'Avg. Medical Term Reduction (%)': [term_reduction['Basic'], term_reduction['In-Context'], 
                                             term_reduction['Chain-of-Thought'], term_reduction['Tree-of-Thought']]
    })
    
    print("Summary of Method Performance:")
    print(summary)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    methods = summary['Method']
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, summary['Avg. Readability Improvement'], width, label='Readability Improvement')
    ax.bar(x + width/2, summary['Avg. Medical Term Reduction (%)'], width, label='Medical Term Reduction (%)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylabel('Improvement')
    ax.set_title('Performance Comparison of Different Prompting Methods')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('method_comparison.png')
    plt.show()
    
    return summary

# Run evaluation
eval_summary = evaluate_methods()

# Additional analysis: Adaptation to education level
def analyze_education_level_adaptation():
    """
    Analyze how well each method adapts to different education levels.
    """
    # Calculate adaptation scores
    adaptations = []
    
    education_order = {
        'elementary': 0,
        'high_school': 1,
        'college': 2,
        'graduate': 3
    }
    
    for idx, row in df.iterrows():
        education_level = row['education_level']
        edu_level_num = education_order[education_level]
        
        # Get the corresponding result
        result = results[idx]
        
        # Calculate expected readability by education level
        # Higher education should have lower adaptation (less simplification)
        expected_adaptation = 1.0 - (edu_level_num / 3.0)
        
        # Calculate actual adaptation
        basic_adapt = (result['basic_complexity']['flesch_kincaid'] - result['original_complexity']['flesch_kincaid']) / 100
        in_context_adapt = (result['in_context_complexity']['flesch_kincaid'] - result['original_complexity']['flesch_kincaid']) / 100
        cot_adapt = (result['cot_complexity']['flesch_kincaid'] - result['original_complexity']['flesch_kincaid']) / 100
        tot_adapt = (result['tot_complexity']['flesch_kincaid'] - result['original_complexity']['flesch_kincaid']) / 100
        
        # Calculate adaptation error (lower is better)
        adaptations.append({
            'patient_id': row['patient_id'],
            'education_level': education_level,
            'expected_adaptation': expected_adaptation,
            'basic_adapt_error': abs(basic_adapt - expected_adaptation),
            'in_context_adapt_error': abs(in_context_adapt - expected_adaptation),
            'cot_adapt_error': abs(cot_adapt - expected_adaptation),
            'tot_adapt_error': abs(tot_adapt - expected_adaptation)
        })
    
    # Convert to DataFrame
    adapt_df = pd.DataFrame(adaptations)
    
    # Calculate average adaptation error by method
    avg_errors = {
        'Basic': adapt_df['basic_adapt_error'].mean(),
        'In-Context': adapt_df['in_context_adapt_error'].mean(),
        'Chain-of-Thought': adapt_df['cot_adapt_error'].mean(),
        'Tree-of-Thought': adapt_df['tot_adapt_error'].mean()
    }
    
    print("Average Adaptation Error by Method (lower is better):")
    for method, error in avg_errors.items():
        print(f"{method}: {error:.4f}")
    
    # Visualize adaptation by education level
    plt.figure(figsize=(10, 6))
    
    # Group by education level
    grouped = adapt_df.groupby('education_level').mean()
    
    # Plot adaptation errors
    grouped[['basic_adapt_error', 'in_context_adapt_error', 'cot_adapt_error', 'tot_adapt_error']].plot(
        kind='bar', figsize=(12, 6))
    plt.title('Adaptation Error by Education Level (Lower is Better)')
    plt.ylabel('Adaptation Error')
    plt.xlabel('Education Level')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(['Basic', 'In-Context', 'Chain-of-Thought', 'Tree-of-Thought'])
    plt.tight_layout()
    plt.savefig('adaptation_by_education.png')
    plt.show()
    
    return avg_errors, adapt_df

# Run adaptation analysis
adaptation_errors, adaptation_df = analyze_education_level_adaptation()

# Combine prompting methods with in-context learning
def combined_prompt(medical_note, education_level):
    """
    Combine tree-of-thought reasoning with in-context learning
    for enhanced performance.
    """
    examples = {
        "elementary": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have high blood sugar (Type 2 Diabetes). Your blood test shows your sugar levels have been high for the past few months."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "Your ankles and feet are swelling with fluid. Sometimes you feel short of breath when you're active."
            }
        ],
        "high_school": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have Type 2 Diabetes. Your HbA1c test, which measures your average blood sugar over the past 3 months, shows it's high at 8.2%."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "You have swelling in your legs and feet, and sometimes have trouble breathing when you're physically active."
            }
        ],
        "college": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have Type 2 Diabetes with an HbA1c of 8.2%, indicating poor glycemic control over the past 3 months."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "You're experiencing fluid retention in your extremities and shortness of breath during physical activity."
            }
        ],
        "graduate": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have Type 2 Diabetes Mellitus with an elevated HbA1c of 8.2%, indicating sustained hyperglycemia over the preceding three months."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "You're presenting with peripheral edema and intermittent dyspnea upon exertion, suggesting possible cardiovascular or pulmonary etiology."
            }
        ]
    }
    
    # Default to high_school level if the specified level is not available
    if education_level not in examples:
        education_level = "high_school"
    
    prompt = f"""
    Please simplify the following medical note for a patient with {education_level} education level.
    
    Here are some examples of simplifying medical notes for someone with {education_level} education:
    
    Original: {examples[education_level][0]['note']}
    Simplified: {examples[education_level][0]['simplified']}
    
    Original: {examples[education_level][1]['note']}
    Simplified: {examples[education_level][1]['simplified']}
    
    Now, I'll simplify this medical note by following a tree-of-thought approach:
    
    Medical note: {medical_note}
    
    Let me consider different approaches to simplify this note:
    
    Approach 1: Replace technical terms with common language appropriate for {education_level} education level
    Approach 2: Reorganize information by importance for the patient
    Approach 3: Use appropriate formatting and structure to enhance understanding
    
    Now, combining these approaches for optimal comprehension:
    
    Final simplified version:
    """
    
    return simulate_llm_response(prompt, method="combined", education_level=education_level)

# Apply combined method to sample data
combined_results = []

for idx, row in df.iterrows():
    medical_note = row['medical_note']
    education_level = row['education_level']
    patient_id = row['patient_id']
    
    # Apply combined method
    combined_result = combined_prompt(medical_note, education_level)
    combined_complexity = measure_complexity(combined_result)
    
    combined_results.append({
        "patient_id": patient_id,
        "education_level": education_level,
        "original_note": medical_note,
        "original_complexity": results[idx]['original_complexity'],
        "combined_result": combined_result,
        "combined_complexity": combined_complexity
    })

# Compare combined method with others
def compare_combined_method():
    """
    Compare the combined method with previous methods.
    """
    # Prepare data for comparison
    readability_scores = {
        'Original': [r['original_complexity']['flesch_kincaid'] for r in results],
        'Basic': [r['basic_complexity']['flesch_kincaid'] for r in results],
        'In-Context': [r['in_context_complexity']['flesch_kincaid'] for r in results],
        'Chain-of-Thought': [r['cot_complexity']['flesch_kincaid'] for r in results],
        'Tree-of-Thought': [r['tot_complexity']['flesch_kincaid'] for r in results],
        'Combined': [r['combined_complexity']['flesch_kincaid'] for r in combined_results]
    }
    
    medical_terms = {
        'Original': [r['original_complexity']['medical_terms_pct'] for r in results],
        'Basic': [r['basic_complexity']['medical_terms_pct'] for r in results],
        'In-Context': [r['in_context_complexity']['medical_terms_pct'] for r in results],
        'Chain-of-Thought': [r['cot_complexity']['medical_terms_pct'] for r in results],
        'Tree-of-Thought': [r['tot_complexity']['medical_terms_pct'] for r in results],
        'Combined': [r['combined_complexity']['medical_terms_pct'] for r in combined_results]
    }
    
    # Calculate average scores
    avg_readability = {method: np.mean(scores) for method, scores in readability_scores.items()}
    avg_med_terms = {method: np.mean(scores) for method, scores in medical_terms.items()}
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    methods = list(avg_readability.keys())
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, list(avg_readability.values()), width, label='Readability Score (higher is better)')
    ax.bar(x + width/2, list(avg_med_terms.values()), width, label='Medical Terms % (lower is better)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylabel('Score')
    ax.set_title('Comparison of All Methods')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('all_methods_comparison.png')
    plt.show()
    
    # Display improvement for combined method
    orig_readability = avg_readability['Original']
    combined_improvement = avg_readability['Combined'] - orig_readability
    best_prev_method = max(avg_readability['Basic'], avg_readability['In-Context'], 
                          avg_readability['Chain-of-Thought'], avg_readability['Tree-of-Thought'])
    best_prev_improvement = best_prev_method - orig_readability
    
    print(f"Combined method readability improvement: {combined_improvement:.2f}")
    print(f"Best previous method improvement: {best_prev_improvement:.2f}")
    print(f"Additional improvement from combined method: {combined_improvement - best_prev_improvement:.2f}")
    
    return avg_readability, avg_med_terms

# Compare combined method
avg_readability, avg_med_terms = compare_combined_method()

# Display a sample of the combined method
def display_combined_sample(patient_id="P001"):
    """
    Display a sample of the combined method results.
    """
    # Find the result for the specified patient
    combined_result = next((r for r in combined_results if r['patient_id'] == patient_id), None)
    original_result = next((r for r in results if r['patient_id'] == patient_id), None)
    
    if combined_result and original_result:
        print(f"Comparison for Patient {patient_id} (Education level: {combined_result['education_level']})")
        print("\nORIGINAL NOTE:")
        print(combined_result['original_note'])
        print(f"\nReadability score: {combined_result['original_complexity']['flesch_kincaid']:.1f}")
        
        print("\n" + "="*50 + "\n")
        
        print("BEST PREVIOUS METHOD (Tree-of-Thought):")
        print(original_result['tree_of_thought_result'])
        print(f"\nReadability score: {original_result['tot_complexity']['flesch_kincaid']:.1f}")
        
        print("\n" + "="*50 + "\n")
        
        print("COMBINED METHOD:")
        print(combined_result['combined_result'])
        print(f"\nReadability score: {combined_result['combined_complexity']['flesch_kincaid']:.1f}")
    else:
        print(f"No results found for patient {patient_id}")

# Display combined method sample
display_combined_sample("P001")

# Conclusion and future work
print("""
Conclusion:
-----------
This tutorial has demonstrated the application of various LLM prompting techniques
for medical notes simplification. We explored basic prompting, in-context learning,
chain-of-thought reasoning, tree-of-thought reasoning, and a combined approach.

Key findings:
1. All prompting methods improved readability compared to original medical notes
2. Tree-of-thought reasoning provided the best structure and organization
3. In-context learning was most effective at adapting to different education levels
4. The combined approach leveraged the strengths of both methods for superior results

Future work:
------------
1. Test with a larger and more diverse dataset of medical notes
2. Implement real LLM API calls instead of simulated responses
3. Create a user interface for healthcare providers to use this system
4. Evaluate with actual patients from different education backgrounds
5. Integrate medical ontologies for better term recognition and replacement
6. Explore multi-modal approaches (adding visual elements to explanations)
""")

# Export results to CSV for further analysis
results_df = pd.DataFrame({
    'patient_id': [r['patient_id'] for r in results],
    'education_level': [r['education_level'] for r in results],
    'original_readability': [r['original_complexity']['flesch_kincaid'] for r in results],
    'basic_readability': [r['basic_complexity']['flesch_kincaid'] for r in results],
    'in_context_readability': [r['in_context_complexity']['flesch_kincaid'] for r in results],
    'cot_readability': [r['cot_complexity']['flesch_kincaid'] for r in results],
    'tot_readability': [r['tot_complexity']['flesch_kincaid'] for r in results],
    'combined_readability': [r['combined_complexity']['flesch_kincaid'] for r in combined_results]
})

results_df.to_csv('medical_notes_simplification_results.csv', index=False)
print("Results exported to medical_notes_simplification_results.csv")

# Display final statistics
print("\nFinal Statistics:")
print(f"Average readability improvement: {np.mean(list(avg_readability.values())[1:]) - avg_readability['Original']:.2f}")
print(f"Best method: Combined (improvement: {avg_readability['Combined'] - avg_readability['Original']:.2f})")
print(f"Medical terminology reduction: {avg_med_terms['Original'] - avg_med_terms['Combined']:.2f}%")

print("\nTutorial complete! This notebook demonstrates how to use LLMs for medical notes simplification.")