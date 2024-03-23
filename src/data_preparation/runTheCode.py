from src.data_preparation.Preprocessing import PreprocessingPipeline


def run_preprocessing_pipeline():
    # Specify the paths to your CSV files
    calls_path = 'datasets/calls.csv'
    reasons_path = 'datasets/reasons.csv'
    referrals_path = 'datasets/referrals.csv'

    # Create an instance of PreprocessingPipeline
    pipeline = PreprocessingPipeline(calls_path, reasons_path, referrals_path)

    # Run the preprocessing pipeline
    pipeline.run()

    # Get and return the processed data
    processed_data = pipeline.get_processed_data()
    return processed_data


if __name__ == "__main__":
    final_df = run_preprocessing_pipeline()
    final_df.to_csv('datasets/processed_data.csv', index=False)
