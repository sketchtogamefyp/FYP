import requests
import time

url = "http://localhost:7860/generate"
sketch_path = "./test_dummy_sketch.png"

with open(sketch_path, "rb") as f:
    files = {"sketch": ("sketch.png", f, "image/png")}
    data = {"description": "Mario platform game where player jumps on platforms to reach the goal flag"}
    r = requests.post(url, files=files, data=data)
    print("Generate response:", r.json())
    job_id = r.json().get("job_id")

for _ in range(30):
    status_r = requests.get(f"http://localhost:7860/status/{job_id}").json()
    print("Status:", status_r.get("step"), f"{status_r.get('progress')}%", status_r.get("status"))
    if status_r.get("status") in ["completed", "error"]:
        break
    time.sleep(1)

print("Final Result:", status_r)
