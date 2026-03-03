import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_alvin_hanafie():
    print("\nLoading Alvin Hanafie datasets...")
    
    train = pd.read_csv("alvin_hanafie/train_binary.tsv", sep='\t')
    valid = pd.read_csv("alvin_hanafie/valid_binary.tsv", sep='\t')
    
    df = pd.concat([train, valid], ignore_index=True)
    df = df[['text', 'label']].copy()
    df['source'] = 'alvin_hanafie'
    
    print(f"  Loaded {len(df):,} samples")
    print(f"  Negative: {(df['label'] == 0).sum():,}, Positive: {(df['label'] == 1).sum():,}")
    
    return df


def load_tiktok_shop():
    print("\nLoading TikTok Shop dataset...")
    
    df = pd.read_csv("tiktok_shop/Tiktok Tokopedia Seller Center Reviews.csv")
    
    label_map = {'negative': 0, 'positive': 1}
    df['label'] = df['sentiment'].map(label_map)
    df = df.rename(columns={'content': 'text'})
    df = df[['text', 'label']].copy()
    df['source'] = 'tiktok_shop'
    
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print(f"  Loaded {len(df):,} samples")
    print(f"  Negative: {(df['label'] == 0).sum():,}, Positive: {(df['label'] == 1).sum():,}")
    
    return df


def load_instagram_cyberbullying():
    print("\nLoading Instagram Cyberbullying dataset...")
    
    df = pd.read_csv("deniyulian/dataset_komentar_instagram_cyberbullying.csv")
    
    label_map = {'negative': 0, 'positive': 1}
    df['label'] = df['Sentiment'].map(label_map)
    df = df.rename(columns={'Instagram Comment Text': 'text'})
    df = df[['text', 'label']].copy()
    df['source'] = 'instagram_cyberbullying'
    
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print(f"  Loaded {len(df):,} samples")
    print(f"  Negative: {(df['label'] == 0).sum():,}, Positive: {(df['label'] == 1).sum():,}")
    
    return df


def load_tweet_tv():
    print("\nLoading Tweet TV Show dataset...")
    
    df = pd.read_csv("deniyulian/dataset_tweet_sentimen_tayangan_tv.csv")
    
    label_map = {'negative': 0, 'positive': 1}
    df['label'] = df['Sentiment'].map(label_map)
    df = df.rename(columns={'Text Tweet': 'text'})
    df = df[['text', 'label']].copy()
    df['source'] = 'tweet_tv'
    
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print(f"  Loaded {len(df):,} samples")
    print(f"  Negative: {(df['label'] == 0).sum():,}, Positive: {(df['label'] == 1).sum():,}")
    
    return df


def load_tweet_film():
    print("\nLoading Tweet Film Opinion dataset...")
    
    df = pd.read_csv("deniyulian/dataset_tweet_sentiment_opini_film.csv")
    
    label_map = {'negative': 0, 'positive': 1}
    df['label'] = df['Sentiment'].map(label_map)
    df = df.rename(columns={'Text Tweet': 'text'})
    df = df[['text', 'label']].copy()
    df['source'] = 'tweet_film'
    
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print(f"  Loaded {len(df):,} samples")
    print(f"  Negative: {(df['label'] == 0).sum():,}, Positive: {(df['label'] == 1).sum():,}")
    
    return df


def combine_all_datasets():
    print("=" * 70)
    print("COMBINING ALL DATASETS")
    print("=" * 70)
    
    alvin = load_alvin_hanafie()
    tiktok = load_tiktok_shop()
    instagram = load_instagram_cyberbullying()
    tweet_tv = load_tweet_tv()
    tweet_film = load_tweet_film()
    
    combined = pd.concat([
        alvin,
        tiktok,
        instagram,
        tweet_tv,
        tweet_film
    ], ignore_index=True)
    
    combined['text'] = combined['text'].fillna('')
    combined = combined[combined['text'].str.strip() != ''].copy()
    
    print("\n" + "=" * 70)
    print("COMBINED DATASET SUMMARY")
    print("=" * 70)
    print(f"\nTotal samples: {len(combined):,}")
    
    print("\nLabel distribution:")
    print(f"  Negative (0): {(combined['label'] == 0).sum():,} ({(combined['label'] == 0).sum() / len(combined) * 100:.2f}%)")
    print(f"  Positive (1): {(combined['label'] == 1).sum():,} ({(combined['label'] == 1).sum() / len(combined) * 100:.2f}%)")
    
    print("\nSource distribution:")
    for source, count in combined['source'].value_counts().items():
        print(f"  {source}: {count:,} ({count / len(combined) * 100:.2f}%)")
    
    return combined


def split_dataset(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    print("\n" + "=" * 70)
    print("SPLITTING DATASET")
    print("=" * 70)
    print(f"Split ratio - Train: {train_size:.0%}, Val: {val_size:.0%}, Test: {test_size:.0%}")
    
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    val_ratio = val_size / (train_size + val_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val['label']
    )
    
    print(f"\nTrain set: {len(train):,} samples")
    print(f"  Negative: {(train['label'] == 0).sum():,}, Positive: {(train['label'] == 1).sum():,}")
    
    print(f"\nValidation set: {len(val):,} samples")
    print(f"  Negative: {(val['label'] == 0).sum():,}, Positive: {(val['label'] == 1).sum():,}")
    
    print(f"\nTest set: {len(test):,} samples")
    print(f"  Negative: {(test['label'] == 0).sum():,}, Positive: {(test['label'] == 1).sum():,}")
    
    return train, val, test


def save_datasets(combined_df, train_df=None, val_df=None, test_df=None, output_dir="combined"):
    print("\n" + "=" * 70)
    print("SAVING DATASETS")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    combined_path = os.path.join(output_dir, "combined_all.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved combined dataset: {combined_path} ({len(combined_df):,} samples)")
    
    if train_df is not None:
        train_path = os.path.join(output_dir, "train.csv")
        train_df.to_csv(train_path, index=False)
        print(f"Saved train set: {train_path} ({len(train_df):,} samples)")
    
    if val_df is not None:
        val_path = os.path.join(output_dir, "val.csv")
        val_df.to_csv(val_path, index=False)
        print(f"Saved validation set: {val_path} ({len(val_df):,} samples)")
    
    if test_df is not None:
        test_path = os.path.join(output_dir, "test.csv")
        test_df.to_csv(test_path, index=False)
        print(f"Saved test set: {test_path} ({len(test_df):,} samples)")
    
    print(f"\nAll files saved in: {output_dir}/")


def main():
    combined = combine_all_datasets()
    
    train, val, test = split_dataset(
        combined, 
        train_size=0.7, 
        val_size=0.15, 
        test_size=0.15,
        random_state=42
    )
    
    save_datasets(
        combined_df=combined,
        train_df=train,
        val_df=val,
        test_df=test,
        output_dir="combined"
    )
    
    print("\n" + "=" * 70)
    print("DATASET COMBINATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()