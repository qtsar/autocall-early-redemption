import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset


def get_asset_data(underlyings, date_str):
    filt_unds = []
    spot_prices = []

    # TODO: потом сделать подсос с yahoo finance
    for und in underlyings:
        und_df = pd.read_csv(f'data/underlying/{und}.csv', sep=';',
                             parse_dates=['date'])

        und_df = \
            und_df.loc[(und_df['date'] <= date_str) &
                       (und_df['date'] >= pd.to_datetime(date_str)
                        - DateOffset(months=12))].set_index('date')

        filt_und = und_df[:date_str]
        spot_prices.append(float(und_df.iloc[-1]))
        filt_unds.append(filt_und)

    spot_prices = np.array(spot_prices)

    df = pd.concat(filt_unds, axis=1)

    # log-returns
    daily_returns = np.log(df) - np.log(df).shift(1)
    daily_returns.dropna(axis=0, inplace=True)
    # df.pct_change().dropna(axis=0)
    # monthly_returns = df.resample('M').first().pct_change().dropna(axis=0)

    ann_drift = (df.iloc[-1] / df.iloc[0]) ** (12/12) - 1
    # ann_vol = monthly_returns.std(axis=0) * 12 ** 0.5
    ann_vol = daily_returns.std(axis=0) * np.sqrt(daily_returns.shape[0])

    if len(underlyings) > 1:
        und_corr = daily_returns.corr()

        return spot_prices, ann_vol.to_numpy(), \
            und_corr.to_numpy(), ann_drift.to_numpy()

    return spot_prices[0], ann_vol[0], None, ann_drift[0]


def get_risk_free_rate_data(currency, date_str):
    df = pd.read_csv(f'data/currencies/{currency}.csv', sep=';',
                     parse_dates=['date'], dayfirst=True, index_col=0)

    row = df[:date_str].iloc[-1].dropna() / 100
    tenors = np.array([int(x) for x in row.index]) / 12

    return tenors, row.to_numpy()


def gbm_one_asset(S0, mu, sigma,
                  T, dt, num_simulations=10):

    N = int(T / dt)

    S = np.zeros((N + 1, num_simulations))
    S[0] = S0

    dW = np.random.normal(0.0, 1.0, (N, num_simulations))

    # cumprod?
    for i in range(N):
        S[i + 1] = S[i] * \
                   np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * dW[i])

    S = S[1:, :].T
    S = S = S[:, np.newaxis, :]

    return S


def gbm_multi_asset(S0, mu, sigma, rho,
                    T, dt, num_simulations = 10):

    N = int(T / dt)

    L = np.linalg.cholesky(rho) if rho is not None else np.eye(S0.shape[0])

    S = np.zeros((N+1, S0.shape[0], num_simulations))
    S[0] = S0.reshape(-1, 1)

    dW = np.random.normal(0.0, 1.0, (N, num_simulations, S0.shape[0]))
    dW_correlated = np.matmul(dW, L.T)

    # cumprod?
    for i in range(N):
        it = (mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * dW_correlated[i]
        S[i+1] = S[i] * np.exp(it).T

    S = S[1:, :, :].T
    return S


def get_price_paths(spot_prices, ann_drift, ann_vol,
                     correlations, term, observation_period, num_simulations, method):

    if method == 'stochastic' and type(spot_prices) in (int, float):
        pass
        # generate by heston model

    elif method == 'constant' and type(spot_prices) == np.ndarray:
        price_paths = gbm_multi_asset(spot_prices, ann_drift, ann_vol, correlations,
                                      term, observation_period, num_simulations)

    else:
        # generate with one asset gbm
        price_paths = gbm_one_asset(spot_prices, ann_drift, ann_vol,
                                    term, observation_period, num_simulations)

    return price_paths





