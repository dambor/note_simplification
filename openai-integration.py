import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from IPython.display import display, HTML
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import openai
from dotenv import load_dotenv
import time

# Load environment variables from .env file (create this file with your API key)
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Download NLTK resources if needed
nltk.download('punkt')
nltk.download('stopwords')

# Function to measure text complexity (same as your original)
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
    
    # List of complex medical terms (this would be expanded in a real application)
    medical_terms = ["hypertension", "myocardial", "infarction", "hyperlipidemia",
                    "atherosclerosis", "dyspnea", "arrhythmia", "tachycardia",
                    "bradycardia", "nephropathy", "neuropathy", "retinopathy",
                    "edema", "amlodipine", "atorvastatin", "ECG", "DASH",
                    "diabetes", "mellitus", "polyuria", "polydipsia", "HbA1c",
                    "microalbuminuria", "metformin", "bronchitis", "purulent",
                    "sputum", "afebrile", "pneumonia", "leukocytosis", "azithromycin",
                    "GERD", "retrosternal", "recumbency", "endoscopy", "esophagitis",
                    "omeprazole", "migraine", "photophobia", "phonophobia", "sumatriptan"]
    
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
                "medical_note": "Patient presents with Stage 2 hypertension (BP 162/94). History of myocardial infarction 3 years ago. Currently on amlodipine 5mg daily and atorvastatin 20mg daily. ECG shows left ventricular hypertrophy. Recommend DASH diet and moderate exercise."
            },
            {
                "patient_id": "P002",
                "age": 72,
                "education_level": "elementary",
                "medical_note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%). Presents with polyuria, polydipsia, and fatigue. Fasting plasma glucose 182 mg/dL. Signs of early diabetic nephropathy with microalbuminuria. Started on metformin 500mg BID."
            },
            {
                "patient_id": "P003",
                "age": 45,
                "education_level": "graduate",
                "medical_note": "Patient exhibits acute bronchitis with purulent sputum production and persistent cough for 10 days. Afebrile, with mild wheezing on auscultation. Chest X-ray rules out pneumonia. CBC shows mild leukocytosis. Prescribed azithromycin 500mg for 3 days."
            },
            {
                "patient_id": "P004",
                "age": 58,
                "education_level": "high_school",
                "medical_note": "Patient presenting with symptoms consistent with GERD, including retrosternal burning pain exacerbated by recumbency and large meals. Endoscopy reveals mild esophagitis without H. pylori. Initiated on PPI therapy with omeprazole 20mg daily."
            },
            {
                "patient_id": "P005",
                "age": 33,
                "education_level": "college",
                "medical_note": "Patient presents with migraine without aura, characterized by unilateral throbbing headache, photophobia, and phonophobia for 2 days. Neurological examination normal. Prescribed sumatriptan 50mg PRN with lifestyle modifications."
            }
        ]
        with open(filename, 'w') as f:
            json.dump(synthetic_data, f, indent=2)
        return synthetic_data

# LLM prompt functions using OpenAI API
def call_openai_api(prompt, model="gpt-4o"):
    """
    Call the OpenAI API with the provided prompt
    """
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Wait and retry once in case of rate limiting
        time.sleep(5)
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error on retry: {e}")
            return f"Error occurred: {e}"

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
    
    return call_openai_api(prompt)

def in_context_learning_prompt(medical_note, education_level):
    """
    In-context learning prompt with examples for specific education levels.
    """
    examples = {
        "elementary": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have high blood sugar (Type 2 Diabetes). Your blood test shows your sugar levels have been high for the past 3 months."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "Your ankles and feet are swelling with fluid. Sometimes you feel short of breath when you're active."
            }
        ],
        "high_school": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have Type 2 Diabetes. Your HbA1c test, which measures your average blood sugar over the past 3 months, shows that your level is 8.2%, which is above the target range."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "You have swelling in your legs and feet, and sometimes have trouble breathing when you're physically active."
            }
        ],
        "college": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have Type 2 Diabetes with an HbA1c of 8.2%, indicating poor glycemic control over the past 3 months. The target is usually below 7.0%."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "You're experiencing fluid retention in your extremities and shortness of breath during physical activity, which may indicate cardiovascular or pulmonary issues."
            }
        ],
        "graduate": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have Type 2 Diabetes Mellitus with an elevated HbA1c of 8.2%, indicating sustained hyperglycemia over the past three months and suboptimal glycemic control."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "You're presenting with peripheral edema and intermittent dyspnea upon exertion, suggesting possible cardiovascular insufficiency or fluid retention requiring further assessment."
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
    
    return call_openai_api(prompt)

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
    
    return call_openai_api(prompt)

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
    
    return call_openai_api(prompt)

def combined_prompt(medical_note, education_level):
    """
    Combine tree-of-thought reasoning with in-context learning
    for enhanced performance.
    """
    examples = {
        "elementary": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have high blood sugar (Type 2 Diabetes). Your blood test shows your sugar levels have been high for the past 3 months."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "Your ankles and feet are swelling with fluid. Sometimes you feel short of breath when you're active."
            }
        ],
        "high_school": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have Type 2 Diabetes. Your HbA1c test, which measures your average blood sugar over the past 3 months, shows that your level is 8.2%, which is above the target range."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "You have swelling in your legs and feet, and sometimes have trouble breathing when you're physically active."
            }
        ],
        "college": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have Type 2 Diabetes with an HbA1c of 8.2%, indicating poor glycemic control over the past 3 months. The target is usually below 7.0%."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "You're experiencing fluid retention in your extremities and shortness of breath during physical activity, which may indicate cardiovascular or pulmonary issues."
            }
        ],
        "graduate": [
            {
                "note": "Patient diagnosed with Type 2 Diabetes Mellitus (HbA1c 8.2%).",
                "simplified": "You have Type 2 Diabetes Mellitus with an elevated HbA1c of 8.2%, indicating sustained hyperglycemia over the past three months and suboptimal glycemic control."
            },
            {
                "note": "Exhibits peripheral edema and occasional dyspnea on exertion.",
                "simplified": "You're presenting with peripheral edema and intermittent dyspnea upon exertion, suggesting possible cardiovascular insufficiency or fluid retention requiring further assessment."
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
    
    return call_openai_api(prompt)

def main():
    # Load the data
    print("Loading sample data...")
    sample_data = load_sample_data()
    df = pd.DataFrame(sample_data)
    print(f"Loaded {len(df)} synthetic medical notes")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables or .env file.")
        print("Please set your OpenAI API key before running this script.")
        return
    
    print("\nProcessing medical notes with OpenAI API...")
    results = []
    
    # Process a sample of the data to save API calls during testing
    # Adjust sample_size as needed
    sample_size = min(2, len(df))
    sample_df = df.iloc[:sample_size]
    
    for idx, row in sample_df.iterrows():
        print(f"\nProcessing patient {row['patient_id']} ({idx+1}/{sample_size})...")
        medical_note = row['medical_note']
        education_level = row['education_level']
        patient_id = row['patient_id']
        
        print(f"  Measuring complexity of original note...")
        original_complexity = measure_complexity(medical_note)
        
        print(f"  Applying basic prompt...")
        basic_result = basic_prompt(medical_note)
        basic_complexity = measure_complexity(basic_result)
        
        print(f"  Applying in-context learning prompt...")
        in_context_result = in_context_learning_prompt(medical_note, education_level)
        in_context_complexity = measure_complexity(in_context_result)
        
        print(f"  Applying chain-of-thought prompt...")
        cot_result = chain_of_thought_prompt(medical_note, education_level)
        cot_complexity = measure_complexity(cot_result)
        
        print(f"  Applying tree-of-thought prompt...")
        tot_result = tree_of_thought_prompt(medical_note, education_level)
        tot_complexity = measure_complexity(tot_result)
        
        print(f"  Applying combined prompt...")
        combined_result = combined_prompt(medical_note, education_level)
        combined_complexity = measure_complexity(combined_result)
        
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
            "tot_complexity": tot_complexity,
            "combined_result": combined_result,
            "combined_complexity": combined_complexity
        })
    
    # Save results
    print("\nSaving results...")
    with open('medical_notes_simplification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display sample comparison for the first patient
    if results:
        first_patient = results[0]
        print(f"\n{'='*50}")
        print(f"Sample comparison for Patient {first_patient['patient_id']} (Education level: {first_patient['education_level']})")
        print(f"\nORIGINAL NOTE:")
        print(first_patient['original_note'])
        print(f"\nReadability score: {first_patient['original_complexity']['flesch_kincaid']:.1f}")
        
        print(f"\n{'='*50}")
        print(f"\nBASIC PROMPT RESULT:")
        print(first_patient['basic_prompt_result'])
        print(f"\nReadability score: {first_patient['basic_complexity']['flesch_kincaid']:.1f}")
        
        print(f"\n{'='*50}")
        print(f"\nTREE-OF-THOUGHT RESULT:")
        print(first_patient['tree_of_thought_result'])
        print(f"\nReadability score: {first_patient['tot_complexity']['flesch_kincaid']:.1f}")
        
        print(f"\n{'='*50}")
        print(f"\nCOMBINED PROMPT RESULT:")
        print(first_patient['combined_result'])
        print(f"\nReadability score: {first_patient['combined_complexity']['flesch_kincaid']:.1f}")
    
    print("\nDone! Results saved to medical_notes_simplification_results.json")

if __name__ == "__main__":
    main()
