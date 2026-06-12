import re

class MedicalSafetyBridge:
    def __init__(self):
        # Advanced Intent Engine: Requires context, not just single words
        self.emergency_intents = [
            r"(heavy|heavily|severe|uncontrollable|profuse|won'?t stop).*(bleed|blood|hemorrhage)",
            r"(chest|heart).*(pain|hurt|crushing|tight|pressure|attack)",
            r"(can'?t|trouble|difficulty|hard to).*(breathe|breathing|catch breath)",
            r"(pass out|passed out|faint|unconscious|black out|seizure)",
            r"(poison|overdose|suicide|drink.*cleaner)"
        ]
        self.medical_keywords = {
            'symptoms': ['pain', 'fever', 'headache', 'nausea', 'vomiting', 'bleed', 'blood', 'dizzy', 'fatigue'],
            'medications': ['dose', 'medication', 'prescription', 'pill', 'mg', 'tablet'],
            'diagnosis': ['diagnose', 'what do i have', 'is it', 'do i have']
        }

    def check_safety(self, user_message: str):
        message_lower = user_message.lower()

        # 1. HARD KILL SWITCH: Check for critical emergencies first
        for pattern in self.emergency_intents:
            if re.search(pattern, message_lower):
                return "kill", "🚨 **CRITICAL EMERGENCY DETECTED** 🚨\n\nYour symptoms indicate a potentially life-threatening emergency. Please stop this chat immediately and call emergency services (102, 108, or 911) or go to the nearest Emergency Room.\n\n*For your safety, the AI diagnostic assistant has been disabled for this prompt to prevent any dangerous delays in care.*"

        # 2. SOFT WARNING: Check for general medical queries
        is_med = any(any(k in message_lower for k in v) for v in self.medical_keywords.values())
        if is_med:
            prefix = "⚠️ **Medical Disclaimer**: I'm not a doctor, and this information is for educational purposes only.\n\n"
            suffix = "\n\n---\n💡 **Please consult a healthcare professional for proper evaluation and treatment.**"
            return "warn", (prefix, suffix)

        # 3. PASS: Normal conversation (like B.Tech projects)
        return "pass", None
