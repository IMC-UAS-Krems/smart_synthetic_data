
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
import requests
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import MinMaxScaler

URL = "https://raw.githubusercontent.com/PacktPublishing/Learning-Pandas-Second-Edition/master/data/goog.csv"
s = requests.get(URL, timeout=5).content
df = pd.read_csv(io.StringIO(s.decode("utf-8")))
df = pd.DataFrame(df.values[::-1], columns=df.columns)

T = (
    pd.to_datetime(df["Date"], infer_datetime_format=True)
    .astype(np.int64)
    .astype(np.float64)
    / 10**9
)
T = pd.Series(MinMaxScaler().fit_transform(T.values.reshape(-1, 1)).squeeze())
df = df.drop(columns=["Date"])

df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
# Build dataset
dataX = []
dataT = []
outcome = []

seq_len = 10

# Cut data by sequence length
for i in range(0, len(df) - seq_len - 1):
    df_seq = df.loc[i : i + seq_len - 1]
    horizons = T.loc[i : i + seq_len - 1]
    out = df["Open"].loc[i + seq_len]

    dataX.append(df_seq)
    dataT.append(horizons.values.tolist())
    outcome.append(out)

# Mix Data (to make it similar to i.i.d)
idx = np.random.permutation(len(dataX))

temporal_data = []
observation_times = []
for i in range(len(dataX)):
    temporal_data.append(dataX[idx[i]])
    observation_times.append(dataT[idx[i]])

outcome = pd.DataFrame(outcome, columns=["Open_next"])
static_data = pd.DataFrame(np.zeros((len(temporal_data), 0)))

loader = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        outcome=outcome,
    )
loader.dataframe()

syn_model = Plugins().get("timegan")

syn_model.fit(loader)

syn_model.generate(count=10).dataframe()

# synthcity absolute
from synthcity.benchmark import Benchmarks

score = Benchmarks.evaluate(
    [
        (f"test_{model}", model, {})
        for model in ["timegan"]
    ],
    loader,
    synthetic_size=1000,
    repeats=2,
    task_type="time_series",  # time_series_survival or time_series
)

Benchmarks.print(score)
