import pandas as pd
from pathlib import Path

results_path = Path("outputs/results_local.csv")
df = pd.read_csv(results_path)

# Convert to a dictionary where the key is the column header, and the value is
# a list of tuples
results_dict = {
    col: df[col].dropna().apply(eval).tolist() for col in df.columns
}

max_rewards = []
for key in results_dict:
    max_reward = max(results_dict[key], key=lambda x: x[2])[2]
    print(f"Max reward for {key}: {max_reward}")
    
    max_rewards.append(max_reward)

print(f"Mean max reward: {sum(max_rewards) / len(max_rewards)}")
