import pandas as pd
import os
import re

def clean_rating(rating_text):
    """Extract numerical rating from text like 'Rated 1 out of 5 stars'."""
    match = re.search(r"Rated (\d+) out of 5 stars", str(rating_text))
    return int(match.group(1)) if match else None

def inspect_dataset(file_path, output_folder):
    try:
        # Load dataset with better error handling
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            engine="python",  # Allows better handling of malformed files
            on_bad_lines="skip"  # Skips corrupted rows instead of throwing an error
        )
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Convert date columns to datetime format
    date_cols = ["Review Date", "Date of Experience"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Clean the 'Rating' column
    if "Rating" in df.columns:
        df["Rating"] = df["Rating"].astype(str).apply(clean_rating)

    # Check for duplicates
    duplicate_count = df.duplicated().sum()

    # Save dataset info
    with open(f"{output_folder}/dataset_info.txt", "w", encoding="utf-8") as f:
        f.write("Dataset Info:\n")
        df.info(buf=f)
        f.write(f"\n\nNumber of duplicate rows: {duplicate_count}")

    # Save the first few rows
    df.head().to_csv(f"{output_folder}/dataset_head.csv", index=False, encoding="utf-8")

    # Save missing values info
    with open(f"{output_folder}/missing_values.txt", "w", encoding="utf-8") as f:
        f.write("Missing values in each column:\n")
        f.write(df.isnull().sum().to_string())

    # Save unique ratings
    if "Rating" in df.columns:
        with open(f"{output_folder}/unique_ratings.txt", "w", encoding="utf-8") as f:
            f.write("Unique values in 'Rating':\n")
            f.write(str(df["Rating"].dropna().unique()))

    # Save summary statistics
    df.describe(include="all").to_csv(f"{output_folder}/dataset_summary.csv", encoding="utf-8")

    print(f"Inspection report saved to {output_folder}")

if __name__ == "__main__":
    file_path = "data/balanced_train.csv"  # Path to your dataset
    output_folder = "results/inspect_after"  # Folder for inspection results
    inspect_dataset(file_path, output_folder)
