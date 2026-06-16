import subprocess
import time
import requests
import sys

def safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            print(msg.encode('ascii', errors='replace').decode('ascii'))
        except Exception:
            pass

def main():
    print("--> Starting uvicorn server on localhost:8000...")
    # Start the server as a background process
    server_process = subprocess.Popen(
        [r".venv\Scripts\uvicorn.exe", "src.app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Wait for the health check to respond
        health_url = "http://127.0.0.1:8000/health"
        max_retries = 20
        connected = False
        print("--> Waiting for server to start...")
        for i in range(max_retries):
            try:
                r = requests.get(health_url, timeout=2)
                if r.status_code == 200:
                    connected = True
                    print("--> Server is up and running!")
                    break
            except requests.RequestException:
                pass
            time.sleep(1.5)
            
        if not connected:
            print("ERROR: Server failed to start in time.", file=sys.stderr)
            # Print stderr to help diagnose
            stderr_output = server_process.stderr.read()
            print(f"Server stderr:\n{stderr_output}", file=sys.stderr)
            sys.exit(1)
            
        # Create a session to maintain cookies
        session = requests.Session()
        
        # Register user
        register_url = "http://127.0.0.1:8000/register"
        register_data = {
            "username": "testuser_langtest",
            "password": "Password123!",
            "confirm_password": "Password123!"
        }
        print("--> Registering test user...")
        session.post(register_url, data=register_data)
        
        # Log in
        login_url = "http://127.0.0.1:8000/login"
        login_data = {
            "username": "testuser_langtest",
            "password": "Password123!"
        }
        print("--> Logging in...")
        login_res = session.post(login_url, data=login_data)
        
        # Send query
        chat_url = "http://127.0.0.1:8000/chat"
        queries_to_test = [
            "i am sad i don't want to kill myself",
            "i am sad",
            "i am depressed",
            "i feel depressed"
        ]
        
        for q in queries_to_test:
            payload = {
                "query": q,
                "history": []
            }
            print(f"--> Sending chat query: '{q}'")
            response = session.post(chat_url, json=payload)
            
            if response.status_code != 200:
                print(f"ERROR: Chat endpoint returned status {response.status_code}", file=sys.stderr)
                print(response.text, file=sys.stderr)
                sys.exit(1)
                
            res_json = response.json()
            safe_print("\n=== RESPONSE FROM ENDPOINT ===")
            safe_print(f"Query:    {q}")
            safe_print(f"Answer:   {res_json.get('answer')}")
            safe_print(f"Language: {res_json.get('language')}")
            safe_print(f"Emotion:  {res_json.get('emotion')}")
            safe_print(f"Intent:   {res_json.get('intent')}")
            safe_print("==============================")
            
            detected_lang = res_json.get('language')
            assert detected_lang == "English", f"Failed: Expected 'English' but got '{detected_lang}'"
        
        print("\nSUCCESS: All queries detected correctly as English!")
        
    finally:
        print("--> Terminating uvicorn server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
            print("--> Server terminated.")
        except subprocess.TimeoutExpired:
            server_process.kill()
            print("--> Server killed.")

if __name__ == "__main__":
    main()
