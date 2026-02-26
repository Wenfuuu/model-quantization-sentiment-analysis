import pandas as pd
import os

def convert_to_binary(input_file, output_file):
    """
    Convert dataset to binary by removing neutral samples
    
    Args:
        input_file: Path to input TSV file
        output_file: Path to output TSV file
    
    Returns:
        DataFrame with binary labels
    """
    # Load dataset
    df = pd.read_csv(input_file, sep='\t')
    
    print(f"\nProcessing: {input_file}")
    print(f"Original shape: {df.shape}")
    print(f"Original sentiment distribution:\n{df['sentiment'].value_counts()}\n")
    
    # Remove neutral samples
    df_binary = df[df['sentiment'] != 'neutral'].copy()
    
    # Map labels: negative=0, positive=1
    label_map = {'negative': 0, 'positive': 1}
    df_binary['label'] = df_binary['sentiment'].map(label_map)
    
    print(f"Binary shape: {df_binary.shape}")
    print(f"Binary sentiment distribution:\n{df_binary['sentiment'].value_counts()}")
    print(f"Binary label distribution:\n{df_binary['label'].value_counts()}\n")
    
    # Save to file
    df_binary.to_csv(output_file, sep='\t', index=False)
    print(f"✅ Saved to: {output_file}")
    
    return df_binary


def main():
    """Process all Alvin Hanafie datasets"""
    
    print("=" * 60)
    print("Converting Alvin Hanafie Dataset to Binary Classification")
    print("=" * 60)
    
    # Define file paths
    files = {
        'train': ('train_preprocess_ori.tsv', 'train_binary.tsv'),
        'valid': ('valid_preprocess.tsv', 'valid_binary.tsv')
    }
    
    # Convert each file
    results = {}
    for split_name, (input_file, output_file) in files.items():
        if os.path.exists(input_file):
            results[split_name] = convert_to_binary(input_file, output_file)
        else:
            print(f"⚠️  File not found: {input_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    
    total_original = sum(len(pd.read_csv(f[0], sep='\t')) for f in files.values() if os.path.exists(f[0]))
    total_binary = sum(len(df) for df in results.values())
    removed = total_original - total_binary
    
    print(f"Total original samples: {total_original:,}")
    print(f"Total binary samples: {total_binary:,}")
    print(f"Neutral samples removed: {removed:,} ({removed/total_original*100:.2f}%)")
    
    print("\nSplit breakdown:")
    for split_name, df in results.items():
        neg = (df['label'] == 0).sum()
        pos = (df['label'] == 1).sum()
        masked = (df['label'] == -1).sum()
        
    print("\n✅ All datasets converted successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
