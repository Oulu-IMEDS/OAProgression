from oaprogression.training import baselines


if __name__ == "__main__":
    args = baselines.init_args()
    train_folds, metadata_test = baselines.init_metadata(args)