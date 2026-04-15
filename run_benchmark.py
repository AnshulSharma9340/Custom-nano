import json
import csv
import requests
import re
import time
import os

# Configuration
DATASET_PATH = "med_benchmark.json" 
RESULTS_PATH = "primum_ai_final_results.csv"
API_URL = "http://localhost:8000/chat/completions"

def ask_model(question, options):
    """Robust format with Chain-of-Thought and strict parsing."""
    
    prompt = f"Question: {question}\n\n"
    for letter, text in options.items():
        prompt += f"{letter}) {text}\n"
    
    # Prompt Engineering: Force a specific, un-confusable prefix
    prompt += "\nBriefly explain your reasoning, then you MUST state your final choice using this exact format: 'FINAL ANSWER: X' where X is A, B, C, or D."

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Strict, analytical mode
        "max_tokens": 100    # Enough room for a brief explanation and the final answer 
    }

    try:
        response = requests.post(API_URL, json=payload, stream=True)
        response.raise_for_status()
        
        full_answer = ""
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str and data_str != '{"done": true}':
                        try:
                            data_json = json.loads(data_str)
                            if "token" in data_json:
                                full_answer += data_json["token"]
                        except json.JSONDecodeError:
                            continue
        
        # Clean the text
        full_answer = full_answer.strip().upper()
        
        # PARSER FIX 1: Look for our strict prefix first (Most reliable)
        strict_match = re.search(r'FINAL ANSWER:\s*([A-D])', full_answer)
        if strict_match:
            return strict_match.group(1)

        # PARSER FIX 2: Look for conversational formats like 'Option C' or 'C)'
        option_match = re.search(r'(?:OPTION|ANSWER IS|CORRECT ANSWER IS)\s*([A-D])|([A-D])\)', full_answer)
        if option_match:
            return option_match.group(1) or option_match.group(2)
        
        # PARSER FIX 3: Look for a standalone letter, but IGNORE it if it's next to Celsius/Degree/F
        letter_match = re.search(r'\b([A-D])\b(?!\s*(?:DEGREE|CELSIUS|FAHRENHEIT|C\b|F\b))', full_answer)
        if letter_match:
            return letter_match.group(1)
            
        return "UNKNOWN"
        
    except Exception as e:
        print(f"API Error: {e}")
        return "ERROR"

def run_benchmark():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Could not find {DATASET_PATH}")
        return

    print("🚀 Starting PrimumAI Robust Benchmark...")
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        questions = json.load(f)
        
    total_questions = len(questions)
    correct_count = 0
    unknown_count = 0
    results = []

    for idx, q in enumerate(questions):
        print(f"Evaluating Question {idx + 1}/{total_questions}...", end=" ", flush=True)
        
        model_answer = ask_model(q["question"], q["options"])
        ground_truth = q["correct_answer"].upper()
        
        if model_answer == "UNKNOWN":
            unknown_count += 1
            print("❓ (Format Not Recognized)")
        else:
            is_correct = (model_answer == ground_truth)
            if is_correct:
                correct_count += 1
                print(f"✅ (Model: {model_answer}, Correct: {ground_truth})")
            else:
                print(f"❌ (Model: {model_answer}, Correct: {ground_truth})")
            
        results.append({
            "id": q.get("id", idx + 1),
            "question": q["question"],
            "expected_answer": ground_truth,
            "model_answer": model_answer,
            "is_correct": model_answer == ground_truth
        })
        
        time.sleep(0.2) # Prevent API overload

    accuracy = (correct_count / total_questions) * 100
    
    print("\n" + "="*45)
    print(f"🏆 Benchmark Complete!")
    print(f"📊 Final Accuracy: {accuracy:.2f}% ({correct_count}/{total_questions})")
    if unknown_count > 0:
        print(f"⚠️ Unparseable Answers: {unknown_count} (Model forgot to state 'A, B, C, or D')")
    print("="*45 + "\n")

    with open(RESULTS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "expected_answer", "model_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"📁 Detailed report saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    run_benchmark()