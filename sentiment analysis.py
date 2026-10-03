# =====================================================================
# TASK 4 — SENTIMENT ANALYSIS
# Classifies text as positive / negative / neutral using TF-IDF
# and compares:
#   1. Multinomial Naive Bayes
#   2. Logistic Regression
#
# Stack:
# pandas, scikit-learn, NLTK, matplotlib, seaborn, wordcloud
#
# DATA:
# Looks for "sentiment_dataset.csv" with columns:
#   text
#   sentiment
#
# If the CSV is not found, a synthetic dataset is generated automatically.
# =====================================================================

import re
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from wordcloud import WordCloud


# =====================================================================
# 1. CONFIGURATION
# =====================================================================

DATA_PATH = "sentiment_dataset.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)


# =====================================================================
# 2. DOWNLOAD REQUIRED NLTK RESOURCES
# =====================================================================

def download_nltk_resources():

    resources = {
        "corpora/stopwords": "stopwords",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4"
    }

    for resource_path, package_name in resources.items():

        try:
            nltk.data.find(resource_path)

        except LookupError:
            print(f"Downloading NLTK resource: {package_name}")
            nltk.download(package_name, quiet=True)


download_nltk_resources()


# =====================================================================
# 3. INITIALIZE NLTK TOOLS
# =====================================================================

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# =====================================================================
# 4. GENERATE SYNTHETIC DATASET
# =====================================================================

def generate_synthetic_sentiment_data(
    n_per_class=500,
    seed=RANDOM_SEED
):

    rng = random.Random(seed)

    subjects = [
        "the product",
        "this app",
        "the service",
        "my order",
        "the delivery",
        "the staff",
        "this phone",
        "the food",
        "the movie",
        "the hotel room",
        "their support team",
        "the laptop",
        "this course",
        "the headphones"
    ]

    positive_templates = [
        "I absolutely love {s}, it exceeded my expectations!",
        "{S} was fantastic and worked perfectly.",
        "Really impressed with {s}, highly recommend it.",
        "{S} is amazing, best purchase I've made this year.",
        "Great experience with {s}, everything went smoothly.",
        "{S} works great and the quality is excellent.",
        "I'm so happy with {s}, will definitely buy again.",
        "{S} is wonderful, exactly what I needed."
    ]

    negative_templates = [
        "I hate {s}, it was a complete waste of money.",
        "{S} was terrible and broke after one day.",
        "Really disappointed with {s}, would not recommend.",
        "{S} is awful, the worst experience I've had.",
        "Bad experience with {s}, nothing worked as promised.",
        "{S} is poor quality and not worth the price.",
        "I'm so frustrated with {s}, it never worked properly.",
        "{S} was not bad, actually it was horrible."
    ]

    neutral_templates = [
        "{S} arrived on Tuesday as scheduled.",
        "I used {s} a few times, it's okay I guess.",
        "{S} does what it says, nothing more nothing less.",
        "{S} comes in a standard box with a manual.",
        "I'm not sure how I feel about {s} yet.",
        "{S} is average, similar to other options out there.",
        "{S} has some pros and some cons, hard to say overall.",
        "{S} was fine, no strong opinion either way."
    ]

    tricky_examples = [
        (
            "Oh great, another broken item, just what I always wanted.",
            "negative"
        ),
        (
            "Well, it's not the worst thing I've ever bought.",
            "neutral"
        ),
        (
            "The screen is beautiful but the battery life is a disaster.",
            "negative"
        ),
        (
            "I guess it's fine, nothing to complain about, nothing to love either.",
            "neutral"
        ),
        (
            "Not amazing, but definitely not bad either.",
            "neutral"
        ),
        (
            "Sure, it works, if you count breaking twice a week as working.",
            "negative"
        ),
        (
            "Honestly it's decent for the price, can't really complain.",
            "positive"
        ),
        (
            "I wanted to love it but it just didn't click for me.",
            "negative"
        ),
        (
            "It does the job, I suppose, though I expected a bit more.",
            "neutral"
        ),
        (
            "Perfect... if you enjoy waiting three weeks for a refund.",
            "negative"
        )
    ]

    rows = []

    bank = {
        "positive": positive_templates,
        "negative": negative_templates,
        "neutral": neutral_templates
    }

    for label, templates in bank.items():

        for _ in range(n_per_class):

            subject = rng.choice(subjects)
            template = rng.choice(templates)

            text = template.format(
                s=subject,
                S=subject.capitalize()
            )

            rows.append({
                "text": text,
                "sentiment": label
            })

    # Add tricky examples
    for text, label in tricky_examples:

        rows.append({
            "text": text,
            "sentiment": label
        })

    df_generated = pd.DataFrame(rows)

    # Add approximately 4% label noise
    noisy_indices = df_generated.sample(
        frac=0.04,
        random_state=seed
    ).index

    classes = [
        "positive",
        "negative",
        "neutral"
    ]

    for index in noisy_indices:

        current_label = df_generated.loc[
            index,
            "sentiment"
        ]

        other_classes = [
            c for c in classes
            if c != current_label
        ]

        df_generated.loc[
            index,
            "sentiment"
        ] = rng.choice(other_classes)

    # Shuffle dataset
    df_generated = df_generated.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    return df_generated


# =====================================================================
# 5. LOAD DATASET
# =====================================================================

try:

    df = pd.read_csv(DATA_PATH)

    print(
        f"Successfully loaded dataset from: {DATA_PATH}"
    )

except (FileNotFoundError, pd.errors.EmptyDataError):

    print(
        f"'{DATA_PATH}' was not found or is empty."
    )

    print(
        "Generating synthetic dataset..."
    )

    df = generate_synthetic_sentiment_data()

    print(
        f"Synthetic dataset generated: {len(df):,} rows"
    )


# =====================================================================
# 6. CHECK REQUIRED COLUMNS
# =====================================================================

required_columns = {
    "text",
    "sentiment"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:

    raise ValueError(
        f"Dataset is missing required columns: "
        f"{', '.join(missing_columns)}\n"
        f"Required columns are: text, sentiment"
    )


# =====================================================================
# 7. CLEAN DATASET
# =====================================================================

df = df[
    ["text", "sentiment"]
].copy()

df = df.dropna(
    subset=["text", "sentiment"]
)

df["text"] = df["text"].astype(str)

df["sentiment"] = (
    df["sentiment"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# Keep only valid sentiment classes
valid_classes = [
    "positive",
    "negative",
    "neutral"
]

df = df[
    df["sentiment"].isin(valid_classes)
].reset_index(drop=True)


# Remove duplicate reviews
df = df.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)


if len(df) == 0:

    raise ValueError(
        "Dataset contains no valid rows after cleaning."
    )


print("\nDataset Preview:")
print(df.head())

print(
    f"\nTotal rows: {len(df):,}"
)


# =====================================================================
# 8. CLASS DISTRIBUTION
# =====================================================================

print("\nClass Distribution:")
print(
    df["sentiment"].value_counts()
)


plt.figure(figsize=(8, 5))

class_order = [
    "positive",
    "negative",
    "neutral"
]

sns.countplot(
    data=df,
    x="sentiment",
    order=[
        c for c in class_order
        if c in df["sentiment"].unique()
    ],
    hue="sentiment",
    palette="Set2",
    legend=False
)

plt.title("Sentiment Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()
plt.show()


# =====================================================================
# 9. TEXT PREPROCESSING
# =====================================================================

def preprocess_text(text):

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove punctuation and numbers
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    try:

        tokens = word_tokenize(text)

    except LookupError:

        nltk.download(
            "punkt",
            quiet=True
        )

        nltk.download(
            "punkt_tab",
            quiet=True
        )

        tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in STOPWORDS
        and len(word) > 1
    ]

    # Lemmatization
    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


df["clean_text"] = df[
    "text"
].apply(preprocess_text)


# Remove rows where preprocessing produced empty text
df = df[
    df["clean_text"].str.strip() != ""
].reset_index(drop=True)


print("\nBefore / After Preprocessing:")

for i in range(min(3, len(df))):

    print(
        f"\nRAW:   {df['text'].iloc[i]}"
    )

    print(
        f"CLEAN: {df['clean_text'].iloc[i]}"
    )


# =====================================================================
# 10. TF-IDF FEATURE EXTRACTION
# =====================================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95
)

X = vectorizer.fit_transform(
    df["clean_text"]
)

y = df["sentiment"]


print(
    f"TF-IDF matrix shape: {X.shape}"
)

print(
    f"Vocabulary size: {len(vectorizer.vocabulary_):,}"
)


# =====================================================================
# 11. TRAIN / TEST SPLIT
# =====================================================================

X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X,
    y,
    df["text"],
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y
)


print(
    f"\nTraining samples: {X_train.shape[0]}"
)

print(
    f"Testing samples: {X_test.shape[0]}"
)


# =====================================================================
# 12. TRAIN MODELS
# =====================================================================

models = {

    "Naive Bayes":
        MultinomialNB(),

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED
        )
}


predictions = {}

trained_models = {}


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    predictions[name] = y_pred

    trained_models[name] = model

    print(
        f"{name} training completed."
    )


# =====================================================================
# 13. MODEL EVALUATION
# =====================================================================

results = []

labels = [
    "negative",
    "neutral",
    "positive"
]


for name, y_pred in predictions.items():

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-score": f1
    })


    print(
        "\n" + "=" * 60
    )

    print(
        f"{name} Classification Report"
    )

    print(
        "=" * 60
    )

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[
                label
                for label in labels
                if label in y_test.unique()
            ],
            zero_division=0
        )
    )


    # Confusion Matrix
    cm_labels = [
        label
        for label in labels
        if label in y.unique()
    ]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=cm_labels
    )


    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=cm_labels,
        yticklabels=cm_labels
    )

    plt.title(
        f"Confusion Matrix — {name}"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.tight_layout()

    plt.show()


# =====================================================================
# 14. MODEL COMPARISON
# =====================================================================

results_df = pd.DataFrame(
    results
)

results_df[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]
] = results_df[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]
].round(3)


print(
    "\n" + "=" * 60
)

print(
    "MODEL COMPARISON"
)

print(
    "=" * 60
)

print(
    results_df.to_string(
        index=False
    )
)


# =====================================================================
# 15. WORD CLOUDS
# =====================================================================

sentiments_present = [
    sentiment
    for sentiment in class_order
    if sentiment in df["sentiment"].unique()
]


fig, axes = plt.subplots(
    1,
    len(sentiments_present),
    figsize=(18, 6)
)


# Handle case where there is only one sentiment
if len(sentiments_present) == 1:

    axes = [axes]


for ax, sentiment in zip(
    axes,
    sentiments_present
):

    text_blob = " ".join(
        df.loc[
            df["sentiment"] == sentiment,
            "clean_text"
        ]
    )

    if text_blob.strip():

        wordcloud = WordCloud(
            width=600,
            height=400,
            background_color="white",
            max_words=80,
            random_state=RANDOM_SEED
        ).generate(text_blob)

        ax.imshow(
            wordcloud,
            interpolation="bilinear"
        )

    ax.set_title(
        f"{sentiment.capitalize()} Reviews",
        fontsize=14
    )

    ax.axis("off")


plt.tight_layout()
plt.show()


# =====================================================================
# 16. ERROR ANALYSIS
# =====================================================================

best_model_name = results_df.loc[
    results_df["F1-score"].idxmax(),
    "Model"
]

best_pred = predictions[
    best_model_name
]


mistakes = pd.DataFrame({

    "text": text_test.values,

    "true_label": y_test.values,

    "predicted_label": best_pred

})


mistakes = mistakes[
    mistakes["true_label"] !=
    mistakes["predicted_label"]
]


print(
    "\n" + "=" * 60
)

print(
    f"ERROR ANALYSIS — {best_model_name}"
)

print(
    "=" * 60
)


if mistakes.empty:

    print(
        "No misclassified examples found."
    )

else:

    sample_size = min(
        5,
        len(mistakes)
    )

    sample_mistakes = mistakes.sample(
        n=sample_size,
        random_state=RANDOM_SEED
    )

    for number, (_, row) in enumerate(
        sample_mistakes.iterrows(),
        start=1
    ):

        print(
            f"\nExample {number}"
        )

        print(
            f"TEXT:      {row['text']}"
        )

        print(
            f"TRUE:      {row['true_label']}"
        )

        print(
            f"PREDICTED: {row['predicted_label']}"
        )


# =====================================================================
# 17. FINAL CONCLUSION
# =====================================================================

best_row = results_df.loc[
    results_df["F1-score"].idxmax()
]


print(
    "\n" + "=" * 60
)

print(
    "CONCLUSION"
)

print(
    "=" * 60
)

print(
    f"Best performing model: "
    f"{best_row['Model']}"
)

print(
    f"Accuracy:  "
    f"{best_row['Accuracy']}"
)

print(
    f"Precision: "
    f"{best_row['Precision']}"
)

print(
    f"Recall:    "
    f"{best_row['Recall']}"
)

print(
    f"F1-score:  "
    f"{best_row['F1-score']}"
)


print(
    """
Real-world applications:

1. Customer feedback classification
   Automatically identify positive, negative and neutral reviews.

2. Social media sentiment analysis
   Analyze public opinions about products or brands.

3. Product review analysis
   Automatically summarize the sentiment of large numbers of reviews.

4. Customer support
   Detect negative messages and prioritize them for faster response.

5. Brand monitoring
   Track changes in customer sentiment over time.
"""
)

print(
    "\nSentiment Analysis Pipeline Completed Successfully!"
)