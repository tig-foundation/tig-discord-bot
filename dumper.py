import requests
import time
import os
import json

API_URL = "https://mainnet-api.tig.foundation"

while True:
    print("Updating latest data...")
    try:
        block = requests.get(f"{API_URL}/get-block?include_data=true").json()['block']
        block_id = block['id']
        height = block['details']['height']
        if os.path.exists(f"{height}.json"):
            print("Already updated")
            print("Sleeping 30s...")
            time.sleep(30)
            continue
        benchmarkers = requests.get(f"{API_URL}/get-opow?block_id={block_id}").json()['opow']
        challenges = requests.get(f"{API_URL}/get-challenges?block_id={block_id}").json()['challenges']
        algorithms = requests.get(f"{API_URL}/get-algorithms?block_id={block_id}").json()['algorithms']
        difficulty = {
            c['id']: requests.get(f"{API_URL}/get-difficulty-data?block_id={block_id}&challenge_id={c['id']}").json()['data']
            for c in challenges
        }
    except Exception as e:
        print(e)
        time.sleep(5)
        continue 
    with open(f"{height}.json", "w") as f:
        json.dump({
            "block": block,
            "benchmarkers": benchmarkers,
            "challenges": challenges,
            "algorithms": algorithms,
            "difficulty": difficulty
        }, f)
    with open(f"latest", "w") as f:
        f.write(str(height))
    if os.path.exists(f"{height - 10080}.json"):
        os.remove(f"{height - 10080}.json")
    print(f"Latest block: (height: {height}, id: {block_id})")
    print("Sleeping 30s...")
    time.sleep(30)