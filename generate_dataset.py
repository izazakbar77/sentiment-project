"""
generate_dataset.py
--------------------
Generates a realistic, balanced synthetic dataset of product reviews
labeled as positive, negative, or neutral.

The dataset mimics real e-commerce review patterns by combining:
- A product category (Electronics, Clothing, Kitchen, Books, Beauty, etc.)
- Sentiment-specific opening/closing phrases
- Sentiment-specific adjectives and complaints/praises
- Randomized review length and structure

This avoids relying on any external dataset download (which keeps the
project fully reproducible offline) while still producing text that is
linguistically realistic enough for a genuine NLP pipeline to learn from.
"""

import random
import pandas as pd

random.seed(42)

PRODUCTS = [
    "phone case", "wireless earbuds", "laptop stand", "running shoes",
    "winter jacket", "blender", "coffee maker", "novel", "face cream",
    "yoga mat", "backpack", "bluetooth speaker", "office chair",
    "kitchen knife set", "smartwatch", "desk lamp", "board game",
    "skincare serum", "gaming mouse", "water bottle", "air fryer",
    "bedsheet set", "sunglasses", "children's toy", "fitness tracker",
    "electric toothbrush", "camera tripod", "hiking boots", "perfume",
    "wall clock"
]

POSITIVE_OPENERS = [
    "I absolutely love this {p}.", "This {p} exceeded my expectations.",
    "Best {p} I have ever bought.", "Honestly impressed with this {p}.",
    "What a fantastic {p}!", "I'm so happy with this {p}.",
    "This {p} is a total game changer.", "Couldn't be more satisfied with this {p}."
]
POSITIVE_BODY = [
    "The quality is excellent and it works perfectly.",
    "It arrived quickly and was packaged really well.",
    "The build quality feels premium and sturdy.",
    "It does exactly what it promises, no complaints at all.",
    "Customer service was also super helpful when I had a question.",
    "The price is very reasonable for the quality you get.",
    "It looks even better in person than in the photos.",
    "Setup was simple and it worked right out of the box.",
    "I've been using it daily and it still performs great.",
    "My whole family loves it now too."
]
POSITIVE_CLOSERS = [
    "Highly recommend it to anyone considering this purchase.",
    "Will definitely buy from this brand again.",
    "Five stars, no hesitation.",
    "Worth every penny.",
    "This is now my go-to recommendation for friends.",
    "I'm extremely satisfied overall."
]

NEGATIVE_OPENERS = [
    "I'm really disappointed with this {p}.", "This {p} was a complete waste of money.",
    "Terrible experience with this {p}.", "I regret buying this {p}.",
    "This {p} broke almost immediately.", "Worst {p} I have purchased in years.",
    "Not happy at all with this {p}.", "This {p} did not meet my expectations at all."
]
NEGATIVE_BODY = [
    "The quality feels cheap and flimsy.",
    "It stopped working after just a few days.",
    "It arrived damaged and customer support was unhelpful.",
    "The description was misleading compared to what I received.",
    "It's much smaller and lower quality than shown in the pictures.",
    "There were missing parts in the package.",
    "It makes a strange noise and doesn't function properly.",
    "Shipping took forever and the packaging was poor.",
    "I tried contacting support but never got a proper response.",
    "It simply doesn't work as advertised."
]
NEGATIVE_CLOSERS = [
    "I would not recommend this to anyone.",
    "Save your money and look elsewhere.",
    "Asking for a refund immediately.",
    "Very frustrating purchase overall.",
    "Two stars at most, and that's being generous.",
    "Extremely disappointed with this purchase."
]

NEUTRAL_OPENERS = [
    "This {p} is okay, nothing special.", "The {p} works as expected, just average.",
    "It's a decent {p} for the price.", "My experience with this {p} was mixed.",
    "This {p} is fine but not amazing.", "An average {p}, does the basic job.",
    "I have a neutral opinion about this {p}.", "The {p} is acceptable overall."
]
NEUTRAL_BODY = [
    "It works fine but doesn't really stand out.",
    "Some features are good while others feel lacking.",
    "The quality is acceptable for the price you pay.",
    "It took a while to arrive but was packaged adequately.",
    "It does the job, though I expected a bit more.",
    "Nothing major to complain about, but nothing exciting either.",
    "It's similar to other products in this category.",
    "A few minor issues but overall usable.",
    "Performance is average compared to similar products.",
    "I might consider other brands next time, but this one is fine."
]
NEUTRAL_CLOSERS = [
    "It's an okay choice if you're on a budget.",
    "Neither great nor terrible, just average.",
    "Three stars, it gets the job done.",
    "Would consider alternatives next time.",
    "Reasonable but not memorable.",
    "Might work for some people, depends on expectations."
]

# Generic, ambiguous sentences that could plausibly appear in reviews of
# almost any sentiment. Mixing these in makes the classification task more
# realistic (real-world reviews are rarely 100% unambiguous).
GENERIC_BODY = [
    "Delivery took about a week.",
    "I bought this as a gift for a family member.",
    "This was my first time buying from this brand.",
    "I compared it with a couple of similar products before buying.",
    "I'll update this review after using it for a longer period.",
    "It's similar in price to other options I looked at.",
    "I saw this product recommended online and decided to try it.",
    "This was a replacement for an older version I used to own.",
]


def build_review(sentiment: str) -> str:
    product = random.choice(PRODUCTS)
    if sentiment == "positive":
        opener = random.choice(POSITIVE_OPENERS).format(p=product)
        body = random.sample(POSITIVE_BODY, k=random.randint(1, 3))
        closer = random.choice(POSITIVE_CLOSERS)
    elif sentiment == "negative":
        opener = random.choice(NEGATIVE_OPENERS).format(p=product)
        body = random.sample(NEGATIVE_BODY, k=random.randint(1, 3))
        closer = random.choice(NEGATIVE_CLOSERS)
    else:
        opener = random.choice(NEUTRAL_OPENERS).format(p=product)
        body = random.sample(NEUTRAL_BODY, k=random.randint(1, 3))
        closer = random.choice(NEUTRAL_CLOSERS)

    # Occasionally mix in a generic/ambiguous sentence (real reviews often
    # contain unrelated filler), making the task less trivially separable.
    if random.random() < 0.35:
        insert_pos = random.randint(0, len(body))
        body.insert(insert_pos, random.choice(GENERIC_BODY))

    parts = [opener] + body + [closer]
    return " ".join(parts), product


def generate_dataset(n_per_class: int = 1200, label_noise: float = 0.06) -> pd.DataFrame:
    rows = []
    for sentiment in ["positive", "negative", "neutral"]:
        for _ in range(n_per_class):
            text, product = build_review(sentiment)
            rows.append({"review_text": text, "product_category": product, "sentiment": sentiment})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle

    # Inject a small amount of realistic label noise (mislabeled / ambiguous
    # reviews happen in real annotated datasets too).
    n_noisy = int(len(df) * label_noise)
    noisy_idx = df.sample(n=n_noisy, random_state=7).index
    all_labels = ["positive", "negative", "neutral"]
    for idx in noisy_idx:
        current = df.at[idx, "sentiment"]
        alt_labels = [l for l in all_labels if l != current]
        df.at[idx, "sentiment"] = random.choice(alt_labels)

    df.insert(0, "review_id", range(1, len(df) + 1))
    return df


if __name__ == "__main__":
    df = generate_dataset(n_per_class=1200, label_noise=0.06)
    df.to_csv("data/product_reviews.csv", index=False)
    print(f"Generated {len(df)} reviews")
    print(df["sentiment"].value_counts())
    print(df.head(3))
