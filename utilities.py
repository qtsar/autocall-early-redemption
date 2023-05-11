import numpy as np
import pandas as pd
from functools import partial
from datetime import date

# from nelson_siegel_svensson.calibrate import calibrate_ns_ols, calibrate_nss_ols
from finance_data import get_asset_data, get_risk_free_rate_data, get_price_paths

freq_dict = {
    'Monthly': 1 / 12,
    'Quarterly': 1 / 4,
    'Semiannually': 1 / 2,
    'Annually': 1.0
}

weighting_dict = {
    'worst': partial(np.min, axis=1),
    'best': partial(np.max, axis=1),
    'average': partial(np.mean, axis=1)
}


def get_all_underlyings(str_path):
    df = pd.read_csv('data/underlyings-available.csv', sep=';')
    df.rename(columns={'long_name': 'label'}, inplace=True)

    sdy = df.set_index('value')['div_yield'].to_dict()
    ls = df[['label', 'value']].sort_values('label').to_dict('records')

    return sdy, ls


SYMBOL_DIV_YIELDS, LABELS_SYMBOLS = get_all_underlyings('data/underlyings-available.csv')


def parse_num_array(input_str):
    res = []
    input_str = input_str.strip()

    for row in input_str.split('\n'):
        row = row.strip()

        if len(row) == 0:
            continue

        if row[-1] in (',', ';'):
            row = row[:-1]

        if ';' in row:
            row = [float(num.strip().replace(',', '.')) if ',' in num else float(num.strip()) for num in row.split(';')]
        else:
            row = [float(num.strip()) for num in row.split(',')]

        res.append(row)

    res = np.array(res)

    if res.shape[0] == 1:
        res = res.flatten()

    if res.shape[0] == 1:
        res = res[0]

    return res


def parse_str_array(input_str):
    input_str = input_str.strip()

    if input_str[-1] in (',', ';'):
        input_str = input_str[:-1]

    if ';' in input_str:
        res = [s.strip().upper() for s in input_str.split(';')]
    else:
        res = [s.strip().upper() for s in input_str.split(',')]

    return res


def validate_inputs(params):
    # PARSE PRODUCT SECTION

    term = float(params['term'] / 12)  # months -> years
    principal = params['principal']

    if params['mode'] == 'Custom':
        spot_prices = parse_num_array(params['spot_prices'])

        if params['dividend_yields'] is None:
            dividend_yields = np.zeros(spot_prices.shape[0]) if type(spot_prices) == np.ndarray else 0.0
        else:
            dividend_yields = parse_num_array(params['dividend_yields']) / 100

        volatilities = parse_num_array(params['volatilities']) / 100

        if params['correlations'] is None:
            correlations = np.eye(spot_prices.shape[0]) if type(spot_prices) == np.ndarray else 1.0
        else:
            correlations = parse_num_array(params['correlations'])

        risk_free_rate_tenors = parse_num_array(params['risk_free_rate_tenors']) / 12
        risk_free_rate_values = parse_num_array(params['risk_free_rate_values']) / 100

        max_rf = min(len(risk_free_rate_tenors), len(risk_free_rate_values))
        risk_free_rate_tenors = risk_free_rate_tenors[:max_rf]
        risk_free_rate_values = risk_free_rate_values[:max_rf]

    else:
        underlyings = params['underlying']  # list

        date_object = date.fromisoformat(params['initial_fixing_date'])  # %Y-%m-%d
        date_string = date_object.strftime('%Y-%m-%d')

        spot_prices, volatilities, correlations, _ = \
            get_asset_data(underlyings, date_string)
        risk_free_rate_tenors, risk_free_rate_values = \
            get_risk_free_rate_data(params['currency'], date_string)

        if len(underlyings) > 1:
            dividend_yields = np.array([SYMBOL_DIV_YIELDS[und] for und in underlyings])
        else:
            dividend_yields = SYMBOL_DIV_YIELDS[(underlyings[0])]

    print(f"spot_prices: {spot_prices}",
          f"dividend_yields: {dividend_yields}",
          f"volatilities: {volatilities}",
          f"correlations: {correlations}",
          f"risk_free_rate_tenors: {risk_free_rate_tenors}",
          f"risk_free_rate_values: {risk_free_rate_values}",
          sep='\n')

    # Nelson Siegel Model Calibration
    # curve_fit, _ = calibrate_nss_ols(risk_free_rate_tenors,
    #                                risk_free_rate_values)

    observation_period = freq_dict[params['observation_freq']]
    observation_dates = np.arange(0, term + observation_period/2, observation_period)

    weighting_method = weighting_dict[params['weighting']]

    # risk_free_rate = curve_fit(observation_dates[1:])  # rates in fractions (as 0.035 = 3.5%)
    risk_free_rate = np.interp(observation_dates[1:],
                               risk_free_rate_tenors,
                               risk_free_rate_values)  # rates in fractions (as 0.035 = 3.5%)

    discount_rate = risk_free_rate  # 0.1% as CDS

    price_paths = get_price_paths(spot_prices=spot_prices,
                                  ann_drift=risk_free_rate[-1] - dividend_yields,
                                  ann_vol=volatilities,
                                  correlations=correlations,
                                  term=term,
                                  observation_period=observation_period,
                                  num_simulations=1_000_000,
                                  method=params['pricing_method'])

    performance = weighting_method(price_paths / spot_prices.reshape(-1, 1))

    observation_dates = observation_dates[1:]

    del price_paths

    # PARSE AUTOCALL SECTION
    print('PARSE AUTOCALL SECTION')

    autocall_conditions = params['autocall_conditions']
    autocall_non_period = params['autocall_non_period'] / 12

    autocall_level = [1.0]
    autocall_payment = 0

    if autocall_conditions == 'fixed':

        autocall_level = params['autocall_level__fixed'] / 100
        autocall_payment = principal

    elif autocall_conditions == 'custom':

        autocall_level = parse_num_array(params['autocall_level__custom']) / 100
        if type(autocall_level) in (float, int):
            autocall_level = np.array([autocall_level])

        autocall_payment = parse_num_array(params['autocall_payment__custom'])

        # print(len(autocall_level), len(autocall_payment))

        autocall_observation_dates = \
            observation_dates[observation_dates >= autocall_non_period]
        autocall_non_observation_dates = \
            observation_dates[observation_dates < autocall_non_period]

        # print(autocall_observation_dates, autocall_non_observation_dates)

        if len(autocall_level) < autocall_observation_dates.shape[0]:
            autocall_level = \
                [999.0] * autocall_non_observation_dates.shape[0] \
                + autocall_level.tolist() + [autocall_level[-1]] * (observation_dates.shape[0] \
                                                                    - autocall_non_observation_dates.shape[0] - len(
                            autocall_level))

            autocall_level = np.array(autocall_level)

        else:
            autocall_level = \
                [999.0] * autocall_non_observation_dates.shape[0] \
                + (autocall_level.tolist())[:autocall_observation_dates.shape[0]]

            autocall_level = np.array(autocall_level)

        if len(autocall_payment) < autocall_observation_dates.shape[0]:
            autocall_payment = \
                [0.0] * autocall_non_observation_dates.shape[0] \
                + autocall_payment.tolist() + [autocall_payment[-1]] * (observation_dates.shape[0] \
                                                                        - autocall_non_observation_dates.shape[0] - len(
                            autocall_payment))

            autocall_payment = np.array(autocall_payment)
        else:
            autocall_payment = \
                [0.0] * autocall_non_observation_dates.shape[0] \
                + (autocall_payment.tolist())[:autocall_observation_dates.shape[0]]

            autocall_payment = np.array(autocall_payment)

        autocall_payment_type = params['autocall_payment_type__custom']
        if autocall_payment_type == '% of Principal':
            autocall_payment *= (principal / 100)

    # print(observation_dates, autocall_level, autocall_payment, autocall_non_period, sep='\n')

    # PARSE COUPON SECTION
    print('PARSE COUPON SECTION')

    coupon_type = params['coupon_type']
    coupon_conditions = params['coupon_conditions']
    coupon_non_period = (params['coupon_non_period'] if params['coupon_non_period'] is not None else 0) / 12
    coupon_paid_at = params['coupon_paid_at']

    coupon_level = 0
    coupon_payment = 0.0

    if coupon_conditions == 'fixed':
        coupon_level = (params['coupon_level__fixed'] if params['coupon_level__fixed'] is not None else 0) / 100
        coupon_payment = (params['coupon_payment__fixed'] if params['coupon_payment__fixed'] is not None else 0)

        coupon_payment_type = params['coupon_payment_type__fixed']

        if coupon_payment_type == '% of Principal':
            coupon_payment *= (principal / 100)
        elif coupon_payment_type == '% of Principal per annum':
            coupon_payment *= (principal / 100 * observation_period)

    elif coupon_conditions == 'custom':
        coupon_level = parse_num_array(params['coupon_level__custom']) / 100
        coupon_payment = parse_num_array(params['coupon_payment__custom'])

        coupon_observation_dates = \
            observation_dates[observation_dates >= coupon_non_period]
        coupon_non_observation_dates = \
            observation_dates[observation_dates < coupon_non_period]

        if len(coupon_level) < coupon_observation_dates.shape[0]:
            coupon_level = \
                [999.9] * coupon_non_observation_dates.shape[0] \
                + coupon_level.tolist() + [coupon_level[-1]] * (observation_dates.shape[0] \
                                                                - coupon_non_observation_dates.shape[0] - len(
                            coupon_level))

            coupon_level = np.array(coupon_level)
        else:
            coupon_level = \
                [999.9] * coupon_non_observation_dates.shape[0] \
                + (coupon_level.tolist())[:coupon_observation_dates.shape[0]]

            coupon_level = np.array(coupon_level)

        if len(coupon_payment) < coupon_observation_dates.shape[0]:
            coupon_payment = \
                [0.0] * coupon_non_observation_dates.shape[0] \
                + coupon_payment.tolist() + [coupon_payment[-1]] * (observation_dates.shape[0] \
                                                                    - coupon_non_observation_dates.shape[0] - len(
                            coupon_payment))

            coupon_payment = np.array(coupon_payment)
        else:
            coupon_payment = \
                [0.0] * coupon_non_observation_dates.shape[0] \
                + (coupon_payment.tolist())[:coupon_observation_dates.shape[0]]

            coupon_payment = np.array(coupon_payment)

        coupon_payment_type = params['coupon_payment_type__custom']

        if coupon_payment_type == '% of Principal':
            coupon_payment *= (principal / 100)
        elif coupon_payment_type == '% of Principal per annum':
            coupon_payment *= (principal / 100 * observation_period)

    # print(observation_dates, coupon_type, coupon_paid_at, coupon_non_period, coupon_level, coupon_payment, sep='\n')

    # PARSE PROTECTION SECTION
    print('PARSE PROTECTION SECTION')

    protection_type = params['protection_type']
    barrier_type = params['barrier_type']
    barrier_level = (params['barrier_level'] if params['barrier_level'] is not None else 0) / 100

    elite_level = (params['elite_level'] if params['elite_level'] is not None else 0) / 100
    elite_payment = (params['elite_payment'] if params['elite_payment'] is not None else 0)
    elite_payment_type = params['elite_payment_type']

    if elite_payment_type == '% of Principal':
        elite_payment *= (principal / 100)

    # print(protection_type, barrier_type, barrier_level, elite_level, elite_payment, sep='\n')

    # calc_bondfloor:
    bondfloor = principal / (1 + discount_rate[-1])**term
    implied_PV = principal

    # commission
    commission = params['commission'] / 100 if params['commission'] is not None else 0.0
    commission_type = params['commission_type'] if params['commission_type'] is not None else 'at once'

    if commission_type == 'per annum':
        implied_PV -= (principal * commission * term)
    else:
        implied_PV -= (principal * commission)

    params_for_calc = [
        principal, discount_rate, performance, observation_dates,
        autocall_level, autocall_payment, autocall_non_period,
        protection_type, barrier_type, barrier_level, elite_level, elite_payment,
        coupon_type, coupon_paid_at, coupon_non_period, coupon_level, coupon_payment,
        implied_PV, params['solve_for']
    ]

    return params_for_calc, np.round(bondfloor, 2)
