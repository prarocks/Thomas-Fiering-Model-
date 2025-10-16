
import numpy as np
import pandas as pd


def thomas_fiering_model(
    historical_file: str,
    sheet_name: str = 'Sheet1',
    num_years: int = 20,
    start_year: int = 2005,
    replace_negative: bool = False,
    random_seed: int | None = None
) -> pd.DataFrame:
   
  """
    Things to Note and Considerations:
    Implements the Thomas-Fiering model for generating synthetic monthly flow data.

    Parameters
    ----------
    historical_file : str
    Path to the Excel file containing historical flow data.         The file must have 12 monthly columns (Jan–Dec).
    sheet_name : str, optional                                      Name of the Excel sheet containing data. Default is 'Sheet1'.
    num_years : int, optional                                       Number of synthetic years to generate. Default is 20.
    start_year : int, optional                                      Starting year for synthetic series. Default is 2005.
    replace_negative : bool, optional                               If True, replaces negative generated flows with 0. Default is False.
    random_seed : int or None, optional                             Random seed for reproducibility. Default is None.

    Returns
    -------
    pd.DataFrame
        Synthetic monthly flow data with years as rows and months as columns.
    """

  
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Load historical data (Jan–Dec columns only)
    df = pd.read_excel(historical_file, sheet_name=sheet_name, usecols=range(1, 13))
    flows = df.values  # Shape: (n_years, 12)
    n_hist = flows.shape[0]

    # Compute means and standard deviations for each month
    means = np.mean(flows, axis=0)
    stds = np.std(flows, axis=0, ddof=1)

    # Compute month-to-month correlations
    corrs = np.zeros(12)
    for j in range(1, 12):
        cov = np.cov(flows[:, j], flows[:, j - 1], ddof=1)[0, 1]
        corrs[j] = cov / (stds[j] * stds[j - 1]) if stds[j] * stds[j - 1] > 0 else 0

    # Correlation between January and previous December
    prev_dec, jan = flows[:-1, 11], flows[1:, 0]
    cov = np.cov(jan, prev_dec, ddof=1)[0, 1]
    corrs[0] = cov / (stds[0] * stds[11]) if stds[0] * stds[11] > 0 else 0

    # Regression coefficients (b_j)
    b = np.zeros(12)
    for j in range(1, 12):
        b[j] = corrs[j] * stds[j] / stds[j - 1]
    b[0] = corrs[0] * stds[0] / stds[11]

    # Random term coefficients (σ_j * sqrt(1 - ρ_j²))
    t_term = stds * np.sqrt(np.maximum(1 - corrs**2, 0))

    # Generate synthetic flows
    synthetic = np.zeros((num_years, 12))
    prev = means[11]  # Initialize with mean December flow

    for i in range(num_years):
        for j in range(12):
            z = np.random.normal(0, 1)
            prev_idx = 11 if j == 0 else j - 1
            synthetic[i, j] = (
                means[j]
                + b[j] * (prev - means[prev_idx])
                + z * t_term[j]
            )
            prev = synthetic[i, j]

    # Replace negative flows if required
    if replace_negative:
        synthetic = np.maximum(synthetic, 0)

    # Format results as DataFrame
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    years = range(start_year, start_year + num_years)
    synthetic_df = pd.DataFrame(synthetic, index=years, columns=months)

    return synthetic_df


if __name__ == "__main__":
    # Example usage
    historical_file = "Test Data.xlsx"
    result = thomas_fiering_model(
        historical_file=historical_file,
        sheet_name="Sheet1",
        num_years=20,
        start_year=2005,
        replace_negative=True,
        random_seed=42
    )
    print("\nSynthetic Flow Data (Thomas-Fiering Model):")
    print(result)
