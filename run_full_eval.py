import json
import requests
import time

# 🔧 CONFIG
API_URL = "http://localhost:8090/chat/completions"
INPUT_FILE = "test_300_questions.jsonl"
OUTPUT_FILE = "model_outputs.jsonl"
DELAY = 0.3
MAX_RETRIES = 3


def query_model(question):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL,
                json={
                    "messages": [
                        {"role": "user", "content": question}
                    ]
                },
                stream=True,   # IMPORTANT for streaming
                timeout=120
            )

            if response.status_code != 200:
                return f"ERROR: status {response.status_code}"

            full_text = ""

            # 🔥 Parse streaming response
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")

                    if decoded.startswith("data: "):
                        try:
                            data = json.loads(decoded.replace("data: ", ""))

                            if "token" in data:
                                full_text += data["token"]

                        except:
                            continue

            return full_text.strip()

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                return f"ERROR: {str(e)}"


def main():
    results = []
    total = 0

    print("🚀 Starting Evaluation...\n")

    with open(INPUT_FILE, "r") as f:
        for line in f:
            item = json.loads(line)
            question = item["question"]

            print(f"[{item['id']}] {item['category']} → {question}")

            answer = query_model(question)

            result = {
                "id": item["id"],
                "category": item["category"],
                "question": question,
                "answer": answer
            }

            results.append(result)
            total += 1

            # Optional: print short preview
            print(f"→ Answer: {answer[:100]}...\n")

            time.sleep(DELAY)

    # Save output
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n✅ Done! {total} questions evaluated")
    print(f"📄 Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()