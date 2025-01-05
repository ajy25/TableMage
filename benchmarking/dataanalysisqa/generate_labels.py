from pathlib import Path
import pandas as pd

curr_dir = Path(__file__).resolve().parent


dataset_name_to_filestem = {
    "Titanic": "titanic",
    "House Prices": "house_prices",
    "Wine": "wine",
    "Breast Cancer": "breast_cancer",
    "Credit": "credit",
    "Abalone": "abalone",
    "Baseball": "baseball",
    "Auto MPG": "autompg",
    "Healthcare": "simulated_healthcare",
    "Iris": "iris",
}

questions_df = pd.read_csv(curr_dir / "dataanalysisqa.tsv", sep="\t")


question_ids = []
labels = []


def generate_labels():
    for dataset_name, filestem in dataset_name_to_filestem.items():
        print(f"Generating labels for {dataset_name}")

        label_script = __import__(f"label_scripts.{filestem}", fromlist=[""])
        labels_dict: dict = label_script.get_labels()

        n_labels = len(labels_dict)

        dataset_df = questions_df[questions_df["Dataset"] == dataset_name]

        for question_order in range(1, n_labels + 1):
            question_id = dataset_df[dataset_df["Question Order"] == question_order][
                "Question ID"
            ].values[0]
            question_ids.append(question_id)
            label = labels_dict[question_order]
            labels.append(label)

    labels_df = pd.DataFrame({"Question ID": question_ids, "Label": labels})
    labels_df.to_csv(curr_dir / "results" / "labels.tsv", sep="\t", index=False)


if __name__ == "__main__":
    generate_labels()
