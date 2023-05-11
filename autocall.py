import numpy as np
from scipy import optimize

PRINCIPAL_BASIS = 10 / 252
COUPON_BASIS = 5 / 252


def get_present_value(principal, discount_rate, performance, observation_dates,
                      autocall_level, autocall_payment, autocall_non_period,
                      protection_type, barrier_type, barrier_level, elite_level, elite_payment,
                      coupon_type, coupon_paid_at, coupon_non_period, coupon_level, coupon_payment):
    """Function to estimate PV of product"""

    print(locals()) # check parameters TODO: remove that line after debug

    # AUTOCALL CONDITIONS

    autocall_condition_1 = ((performance >= autocall_level) * (observation_dates >= autocall_non_period))
    # cumsum to cut off all autocall after first
    autocall_condition_2 = autocall_condition_1.cumsum(axis=1)

    # no autocall event at last date
    autocall_event = (autocall_condition_1 * autocall_condition_2 == 1)
    autocall_event[:, -1] = False

    principal_cf = autocall_event * autocall_payment

    # define scenarios where product is alive till the end
    till_end = ~(autocall_event.any(axis=1))

    # ===================================================================================
    # PROTECTION CONDITIONS
    # redemption_cf = principal_cf[:, -1]

    if protection_type == 'not':
        redemption_cf = np.minimum(performance[:, -1], 1.0) * principal

    elif protection_type == 'full':
        redemption_cf = principal

    # 'one', 'two'
    else:
        barrier_payment = principal

        if barrier_type == 'soft':
            negative_payment = principal * np.minimum(1.0 + performance[:, -1] - barrier_level, 1.0)
        else:
            negative_payment = principal * np.minimum(performance[:, -1], 1.0)

        if protection_type == 'one':
            redemption_cf = (performance[:, -1] >= barrier_level) * barrier_payment \
                + (performance[:, -1] < barrier_level) * negative_payment

        else:
            redemption_cf = (performance[:, -1] >= elite_level) * elite_payment \
                + (performance[:, -1] < elite_level) * (performance[:, -1] >= barrier_level) * barrier_payment \
                + (performance[:, -1] < barrier_level) * negative_payment

    principal_cf[:, -1] = till_end * redemption_cf
    discounted_principal_cf = principal_cf / (1 + discount_rate) ** (observation_dates + PRINCIPAL_BASIS)

    # ===================================================================================
    # COUPON CONDITIONS
    if coupon_type == 'not':
        coupon_cf = 0.0
        discounted_coupon_cf = 0.0

    else:
        # проверяем допустимы ли купонные выплаты
        coupon_active = (autocall_event + (autocall_condition_2 == 0)) * (observation_dates >= coupon_non_period)

        if coupon_type == 'guaranteed':
            coupon_cf = coupon_active * coupon_payment

        elif coupon_type == 'contingent':
            coupon_condition = (performance >= coupon_level)
            coupon_cf = coupon_active * coupon_condition * coupon_payment

        elif coupon_type == 'memory':
            coupon_condition = (performance >= coupon_level)

            arr = (~coupon_condition) * coupon_active
            rr, cc = np.where((arr[:, 1:] == 0) & (arr[:, :-1] != 0))
            reduce_indices = rr * arr.shape[1] + cc + 1

            row_starts = np.arange(arr.shape[0]) * arr.shape[1]
            reduce_indices = np.hstack((row_starts, reduce_indices))
            reduce_indices = np.sort(reduce_indices)

            totals = np.add.reduceat(arr.flatten(), reduce_indices)
            result_f = np.zeros((arr.size,))
            result_f[reduce_indices[1:]] = totals[:-1]
            accum_non_paid = result_f.reshape(arr.shape)
            accum_non_paid[:, 0] = 0

            coupon_cf = (coupon_active * coupon_condition + accum_non_paid) * coupon_payment

        if coupon_paid_at == 'maturity':
            autocall_event[:, -1] = till_end
            coupon_cf = autocall_event * coupon_cf.sum(axis=1, keepdims=True)

        discounted_coupon_cf = coupon_cf / (1 + discount_rate) ** (observation_dates + COUPON_BASIS)

    # ===================================================================================
    # SUM ALL CASH FLOWS
    # all_cf = coupon_cf + principal_cf
    discounted_all_cf = discounted_coupon_cf + discounted_principal_cf

    # GET PV
    summ = discounted_all_cf.sum(axis=1)
    pv_mean = np.mean(summ)
    pv_median = np.median(summ)
    pv = np.round(pv_mean / 2 + pv_median / 2, 2)

    # PROBABILITIES OF AUTOCALL
    autocall_proba = autocall_event.sum(axis=0) / performance.shape[0] * 100

    # PROBABILITIES OF COUPON PAYMENTS
    if coupon_type in ('contingent', 'memory'):
        coupon_proba = (coupon_condition * coupon_active).sum(axis=0) / performance.shape[0] * 100
        # coupon_proba = coupon_condition.sum(axis=0) / performance.shape[0] * 100
        return pv, autocall_proba, coupon_proba

    return pv, autocall_proba, 0


def get_coupon_annual_yield(principal, discount_rate, performance, observation_dates,
                      autocall_level, autocall_payment, autocall_non_period,
                      protection_type, barrier_type, barrier_level, elite_level, elite_payment,
                      coupon_type, coupon_paid_at, coupon_non_period, coupon_level, coupon_payment,
                      implied_PV):
    """Function to estimate PV of product"""

    print(locals()) # check parameters TODO: remove that line after debug

    # AUTOCALL CONDITIONS

    autocall_condition_1 = ((performance >= autocall_level) * (observation_dates >= autocall_non_period))
    # cumsum to cut off all autocall after first
    autocall_condition_2 = autocall_condition_1.cumsum(axis=1)

    # no autocall event at last date
    autocall_event = (autocall_condition_1 * autocall_condition_2 == 1)
    autocall_event[:, -1] = False

    principal_cf = autocall_event * autocall_payment

    # define scenarios where product is alive till the end
    till_end = ~(autocall_event.any(axis=1))

    # ===================================================================================
    # PROTECTION CONDITIONS
    # redemption_cf = principal_cf[:, -1]

    if protection_type == 'not':
        redemption_cf = np.minimum(performance[:, -1], 1.0) * principal

    elif protection_type == 'full':
        redemption_cf = principal

    # 'one', 'two'
    else:
        barrier_payment = principal

        if barrier_type == 'soft':
            negative_payment = principal * np.minimum(1.0 + performance[:, -1] - barrier_level, 1.0)
        else:
            negative_payment = principal * np.minimum(performance[:, -1], 1.0)

        if protection_type == 'one':
            redemption_cf = (performance[:, -1] >= barrier_level) * barrier_payment \
                + (performance[:, -1] < barrier_level) * negative_payment

        else:
            redemption_cf = (performance[:, -1] >= elite_level) * elite_payment \
                + (performance[:, -1] < elite_level) * (performance[:, -1] >= barrier_level) * barrier_payment \
                + (performance[:, -1] < barrier_level) * negative_payment

    principal_cf[:, -1] = till_end * redemption_cf
    discounted_principal_cf = principal_cf / (1 + discount_rate) ** (observation_dates + PRINCIPAL_BASIS)

    # ===================================================================================
    # COUPON CONDITIONS
    # проверяем допустимы ли купонные выплаты
    coupon_active = (autocall_event + (autocall_condition_2 == 0)) * (observation_dates >= coupon_non_period)

    if coupon_type == 'guaranteed':
        coupon_factor = coupon_active

    elif coupon_type == 'contingent':
        coupon_condition = (performance >= coupon_level)
        coupon_factor = coupon_active * coupon_condition

    elif coupon_type == 'memory':
        coupon_condition = (performance >= coupon_level)

        arr = (~coupon_condition) * coupon_active
        rr, cc = np.where((arr[:, 1:] == 0) & (arr[:, :-1] != 0))
        reduce_indices = rr * arr.shape[1] + cc + 1

        row_starts = np.arange(arr.shape[0]) * arr.shape[1]
        reduce_indices = np.hstack((row_starts, reduce_indices))
        reduce_indices = np.sort(reduce_indices)

        totals = np.add.reduceat(arr.flatten(), reduce_indices)
        result_f = np.zeros((arr.size,))
        result_f[reduce_indices[1:]] = totals[:-1]
        accum_non_paid = result_f.reshape(arr.shape)
        accum_non_paid[:, 0] = 0

        coupon_factor = (coupon_active * coupon_condition + accum_non_paid)

    if coupon_paid_at == 'maturity':
        autocall_event[:, -1] = till_end
        coupon_factor = autocall_event * coupon_factor.sum(axis=1, keepdims=True)

    def optim_func(coupon_payment1):
        coupon_cf = coupon_factor * coupon_payment1
        discounted_coupon_cf = coupon_cf / (1 + discount_rate) ** (observation_dates + COUPON_BASIS)

        # SUM ALL CASH FLOWS
        # all_cf = coupon_cf + principal_cf
        discounted_all_cf = discounted_coupon_cf + discounted_principal_cf

        # GET PV
        summ = discounted_all_cf.sum(axis=1)
        pv_mean = np.mean(summ)
        pv_median = np.median(summ)
        pv = np.round(pv_mean / 2 + pv_median / 2, 2)

        return np.abs(implied_PV - pv)

    coupon_payment = optimize.root(optim_func, 1).x[0]
    coupon_payment = {'in money:': np.round(coupon_payment, 2),
                      '% of Principal': np.round(coupon_payment / principal * 100, 2),
                      '% of Principal p.a.': np.round(coupon_payment / principal * 100 / observation_dates[0], 2)
                      }

    # PROBABILITIES OF AUTOCALL
    autocall_proba = autocall_event.sum(axis=0) / performance.shape[0] * 100

    # PROBABILITIES OF COUPON PAYMENTS
    if coupon_type in ('contingent', 'memory'):
        coupon_proba = (coupon_condition * coupon_active).sum(axis=0) / performance.shape[0] * 100
        # coupon_proba = coupon_condition.sum(axis=0) / performance.shape[0] * 100
        return coupon_payment, autocall_proba, coupon_proba

    return coupon_payment, autocall_proba, 0














