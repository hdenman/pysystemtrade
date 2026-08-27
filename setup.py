from __future__ import print_function
modified by omp
modified by omp
import os
import sys
import platform
from setuptools import setup, find_packages
from distutils.version import StrictVersion

######################################################################
# This method of installing the project is deprecated
# Dependencies and package data are defined in 'pyproject.toml'
# See the README
# This file will be removed in a future release
######################################################################


if StrictVersion(platform.python_version()) <= StrictVersion("3.10.0"):
    print("pysystemtrade requires Python 3.10.0 or later. Exiting.", file=sys.stderr)
    sys.exit(1)


def read(fname):
    """Utility function to read the README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def package_files(directory, extension="yaml"):
import argparse
from datetime import date
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            if filename.split(".")[-1] == extension:
                paths.append(os.path.join("..", path, filename))

    return paths


def dir_this_file():
    return os.path.dirname(os.path.realpath(__file__))


private_dir = os.path.join(dir_this_file(), "private")
private_yaml_files = package_files(private_dir, "yaml")

provided_dir = os.path.join(dir_this_file(), "systems", "provided")
provided_yaml_files = package_files(provided_dir, "yaml")

control_dir = os.path.join(dir_this_file(), "syscontrol")
control_yaml_files = package_files(control_dir, "yaml")

data_csv_path = os.path.join(dir_this_file(), "data")
data_csv_files = package_files(data_csv_path, "csv")
def _load_csv_prices(start_date: date = None, end_date: date = None) -> futuresAdjustedPrices:
    csv_file = resolve_path_and_filename_for_package(CSV_PATH)
    raw = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index, utc=False)
    prices = raw.iloc[:, 0]

    # CSV has intraday rows — take last price each business day
    daily = prices.resample("1B").last().dropna()
    daily.index = daily.index.normalize()

    if start_date is not None:
        daily = daily[daily.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        daily = daily[daily.index <= pd.Timestamp(end_date)]

    daily.name = "price"
    return futuresAdjustedPrices(daily)
brokers_csv_files = package_files(brokers_csv_path, "csv")

brokers_yaml_path = os.path.join(dir_this_file(), "sysbrokers")
brokers_yaml_files = package_files(brokers_yaml_path, "yaml")

package_data = {
def write_parquet_prices(start_date: date = None, end_date: date = None) -> None:
    store  = _price_store()
    prices = _load_csv_prices(start_date=start_date, end_date=end_date)
    + test_data_csv_files
    + brokers_csv_files
    + brokers_yaml_files
    + control_yaml_files
    + default_config_yaml_files
}

print(package_data)

setup(
    name="pysystemtrade",
    version="1.8.2",
    author="Robert Carver",
    description=(
        "Python framework for running systems as in Robert Carver's book Systematic Trading"
        " (https://www.systematicmoney.org/systematic-trading)"
    ),
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load SP500 adjusted prices from CSV to parquet store."
    )
    parser.add_argument("--start", type=date.fromisoformat, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"universe : backtest")
    print(f"parquet  : {scoped_path('PARQUET_DATA')}")
    print()

    print(f"Loading {INSTRUMENT_CODE} from CSV …")
    write_parquet_prices(start_date=args.start, end_date=args.end)
        "pandas==2.1.3",
        "matplotlib>=3.0.0",
        "ib_async>=2,<3",
        "PyYAML>=5.3",
        "numpy>=1.24.0",
        "scipy>=1.0.0",
        "pymongo==3.11.3",
        "psutil==5.6.7",
        "pytest>6.2",
        "Flask>=2.0.1",
        "Werkzeug>=2.0.1",
        "statsmodels==0.14.0",
        "PyPDF2>=2.5.0",
        "pyarrow>=14.0.1",
        "scikit-learn>1.3.0",
    ],
    tests_require=["nose", "flake8"],
    extras_require=dict(),
    test_suite="nose.collector",
    include_package_data=True,
)
