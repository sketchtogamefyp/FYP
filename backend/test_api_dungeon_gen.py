import requests
import time

url = "http://localhost:7860/generate"
sketch_path = "./test_dummy_sketch.png"

with open(sketch_path, "rb") as f:
    files = {"sketch": ("sketch.png", f, "image/png")}
    data = {"description": "A dark dungeon combat game where the knight explores dangerous dungeon halls and defeats skeleton warrior"}
    r = requests.post(url, files=files, data=data)
    print("Dungeon generate response:", r.json())
    job_id = r.json().get("job_id")

for _ in range(30):
    status_r = requests.get(f"http://localhost:7860/status/{job_id}").json()
    print("Status:", status_r.get("step"), f"{status_r.get('progress')}%", status_r.get("status"))
    if status_r.get("status") in ["completed", "error"]:
        break
    time.sleep(1)

print("Final Result:", status_r)
