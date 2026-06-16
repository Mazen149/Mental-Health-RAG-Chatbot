import json
import random
import string
from locust import HttpUser, task, between

class SanadUser(HttpUser):
    # Wait between 5 and 15 seconds between tasks to simulate human reading speed
    wait_time = between(5, 15)
    # Default host (override in the Locust UI or via --host)
    host = "http://localhost:8000"

    def on_start(self):
        """Register a unique user account for each virtual user to obtain a session cookie"""
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        self.username = f"locust_{random_suffix}"
        self.password = "testpass"

        payload = {"username": self.username, "password": self.password}
        with self.client.post("/register", json=payload, name="/register", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 400 and "already registered" in response.text:
                self.client.post("/login", json=payload, name="/login")
                response.success()
            else:
                response.failure(f"Auth failed: {response.status_code}")

    @task(3)
    def chat_greeting(self):
        """Simulate a user greeting the chatbot"""
        payload = {
            "message": "Hello, how are you today?"
        }
        self.client.post("/chat", json=payload, name="/chat (Greeting)", timeout=30.0)

    @task(2)
    def chat_mental_health(self):
        """Simulate a user asking a mental health question (triggers RAG)"""
        payload = {
            "message": "I've been feeling extremely anxious lately and having trouble sleeping. What can I do?"
        }
        self.client.post("/chat", json=payload, name="/chat (Mental Health)", timeout=30.0)

    @task(1)
    def check_health(self):
        """Simulate basic health checks"""
        self.client.get("/health", name="/health", timeout=10.0)

    @task(1)
    def submit_feedback(self):
        """Simulate a user giving feedback"""
        payload = {
            "vote": "up",
            "user_message": "Hello, how are you today?",
            "bot_response": "I am here to help you."
        }
        self.client.post("/feedback", json=payload, name="/feedback")
