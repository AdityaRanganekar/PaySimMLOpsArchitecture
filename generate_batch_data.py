import pandas as pd
import os

def create_batch_test_data():
    print("Reading original dataset...")

    df = pd.read_csv("transactions_data/data.csv")
    
    # 10 frauds and 90 normal transactions
    print("Sampling data...")
    frauds = df[df['isFraud'] == 1].sample(10, random_state=42)
    normals = df[df['isFraud'] == 0].sample(90, random_state=42)
    
    # Combine and shuffle
    batch_df = pd.concat([frauds, normals]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Drop the target columns 
    batch_df = batch_df.drop(columns=['isFraud', 'isFlaggedFraud'], errors='ignore')
    
    # Create a directory for incoming batch data
    os.makedirs("valid_data", exist_ok=True)

    file_path = "valid_data/test_batch.csv"
    batch_df.to_csv(file_path, index=False)
    print(f"Success! Created {file_path} with {len(batch_df)} rows.")

if __name__ == "__main__":
    create_batch_test_data()