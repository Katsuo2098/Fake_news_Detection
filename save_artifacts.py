from pathlib import Path
import pickle


BASE_DIR = Path(__file__).resolve().parent


def save_artifacts(trained_model, trained_vectorizer) -> None:
    """Save the trained model and TF-IDF vectorizer in the project root."""
    with (BASE_DIR / "model.pkl").open("wb") as model_file:
        pickle.dump(trained_model, model_file)

    with (BASE_DIR / "vectorizer.pkl").open("wb") as vectorizer_file:
        pickle.dump(trained_vectorizer, vectorizer_file)

    print(f"Saved model to: {BASE_DIR / 'model.pkl'}")
    print(f"Saved vectorizer to: {BASE_DIR / 'vectorizer.pkl'}")


if __name__ == "__main__":
    print(
        "Import save_artifacts(trained_model, trained_vectorizer) in your notebook "
        "after training, then call it with your fitted objects."
    )
