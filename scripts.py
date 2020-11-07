from calc_metrics import calc_metrics
from pathlib import Path


def calc_metrics_dir(path, metric_names, metricdata, mirror=True, gpus=1, reverse=True, first=None):

    """
    calc_metric for networks contained in directory. Optionally, select first networks (or last if reverse is true).
    """

    if not isinstance(metric_names, list):
        metric_names = [metric_names]

    path = Path(path)
    print(path)
    networks = [p for p in path.glob("*.pkl")]
    networks = sorted(networks, reverse=reverse)
    print("total networks in folder: ", len(networks))

    if first:
        assert 1 <= first <= len(networks), f"first must be between 1 and {len(networks)}"
        networks = networks[:first]

    print(f"Calculating metrics {metric_names} for:\n{[n.name for n in networks]}")

    for ns in networks:
        calc_metrics(str(ns), metric_names, metricdata, mirror, gpus)


if __name__ == "__main__":

    path = "/home/alberto/Data/github/stylegan2-ada/models/trv_test"
    data = "/home/alberto/Data/github/stylegan2-ada/data/trv_s128"
    calc_metrics_dir(path, "fid50k_full", data, first=2)
