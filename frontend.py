from dash import Dash, dcc, html, Input, Output, ctx
from utilities import validate_inputs, LABELS_SYMBOLS
from autocall import get_present_value, get_coupon_annual_yield
from datetime import date

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# для options нужно будет подгрузить названия компаний

app.layout = html.Div([

    ### SECTION FOR MODE
    html.Section([
        html.Label('Calculator Mode'),

        dcc.Dropdown(
            options=['Ticker', 'Custom'],
            placeholder='Select the mode',
            value='Ticker',
            id='input--mode'
        ),

        html.Label('Custom: Enter spot prices, volatilities, correlations, risk-free interest rates'),
        html.Label('Ticker: The asset data are retrieved themselves, based on EOD or Yahoo Finance'),
    ]),

    # ===============================================================
    # SECTION FOR PRODUCT INFO
    html.Section([
        html.Label('Product Information'),

        html.Div([
            html.Div([
                html.Label('Underlying'),
                # dcc.Input(type='text',
                #           placeholder="Enter symbols by ';' ",
                #           id='input--underlying')

                dcc.Dropdown(
                    # TODO: take labels and values from files
                    options=LABELS_SYMBOLS,
                    placeholder='Enter underlyings',
                    multi=True,
                    id='input--underlying'
                )
            ]),

            html.Div([
                html.Label('Initial Fixing Date'),
                dcc.DatePickerSingle(
                    min_date_allowed=date(2015, 1, 5),
                    max_date_allowed=date(2023, 5, 1),
                    date=date(2023, 5, 1),
                    id='input--initial-fixing-date'
                )
            ]),

        ],
            id='asset--mode-ticker',
            style={'display': 'none'}  # hide element in custom mode
        ),

        html.Div([
            html.Div([
                html.Label('Spot Prices'),
                dcc.Input(
                    type='text',
                    placeholder="100, 80, ...",
                    id='input--spot-prices'
                )
            ]),

            html.Div([
                html.Label('Annual Dividend yields (%)'),
                dcc.Input(
                    type='text',
                    placeholder="5.0, 6.0 ...",
                    id='input--dividend-yields'
                )
            ]),

            html.Div([
                html.Label('Annual Volatilities (%)'),
                dcc.Input(
                    type='text',
                    placeholder="30, 25, ...",
                    id='input--volatilities'
                )
            ]),

            html.Div([
                html.Label('Correlations'),
                dcc.Textarea(
                    placeholder="1, 0.5 ,...\n0.5, 1, ...\n ...",
                    id='input--correlations'
                )
            ])
        ],
            id='asset--mode-custom',
            style={'display': 'block'}
        ),

        html.Div([
            html.Label('Term (Months)'),
            dcc.Input(
                type='number',
                placeholder="60",
                id='input--term'
            )
        ]),

        html.Div([
            html.Label('Weighting'),
            dcc.Dropdown(
                options=[
                    {'label': 'Worst-of Performance', 'value': 'worst'},
                    {'label': 'Average-of Performance', 'value': 'average'},
                    {'label': 'Best-of Performance', 'value': 'best'}
                ],
                placeholder='Select a weighting',
                value='worst',
                id='input--weighting'
            ),
        ]),

        html.Div([
            html.Label('Observation Dates Frequency'),
            dcc.Dropdown(
                options=[
                    'Monthly',
                    'Quarterly',
                    'Semiannually',
                    'Annually'
                ],
                placeholder='Select a frequency',
                id='input--observation-freq'
            ),
        ]),

        html.Div([
            html.Label('Principal'),
            dcc.Input(
                type='number',
                placeholder="1000",
                id='input--principal',
                value=1000
            )
        ]),

        html.Div([
            html.Label('Currency'),
            dcc.Dropdown(
                options=['USD', 'EUR', 'GBP', 'CHF'],
                placeholder="Select a currency",
                id='input--currency'
            )
        ],
            id='currency--mode-ticker',
            style={'display': 'none'}  # hide element in custom mode
        ),

        html.Div([
            html.Label('Risk Free Rate Tenors (Months)'),
            dcc.Input(
                type='text',
                placeholder="1, 3, 6, ...",
                id='input--risk-free-rate-tenors'
            ),

            html.Label('Risk Free Rate Values (%)'),
            dcc.Input(
                type='text',
                placeholder="1.0, 1.5, 1.75, ...",
                id='input--risk-free-rate-values'
            )
        ],
            id='currency--mode-custom',
            style={'display': 'block'}  # hide element in custom mode
        )
    ]),

    # ===============================================================
    # SECTION FOR AUTOCALL
    html.Section([
        html.Label('Autocall Section'),

        html.Div([
            html.Label('Autocall Conditions'),
            dcc.Dropdown(
                options=[
                    {'label': 'Fixed (Same for all Dates)', 'value': 'fixed'},
                    {'label': 'Custom (Different for Dates)', 'value': 'custom'},
                ],
                placeholder='Select autocall conditions',
                id='input--autocall-conditions'
            ),
        ]),

        html.Div([
            html.Label('Autocall Level (%)'),
            dcc.Input(
                type='number',
                placeholder="100",
                id='input--autocall-level--fixed',
                value=100,
            ),
        ],
            id='autocall-level--conditions-fixed',
            style={'display': 'none'}
        ),

        html.Div([
            html.Label('Autocall Levels (%)'),
            dcc.Input(
                type='text',
                placeholder="100, 97.5, 95, ...",
                id='input--autocall-level--custom'
            ),
        ],
            id='autocall-level--conditions-custom',
            style={'display': 'none'}
        ),

        html.Div([
            html.Label('Autocall Payments'),
            dcc.Input(
                type='text',
                placeholder="100, 97.5, 95, ...",
                id='input--autocall-payment--custom'
            ),
            dcc.RadioItems(
                options=['in money', '% of Principal'],
                inline=True,
                id='input--autocall-payment-type--custom'
            )
        ],
            id='autocall-payment--conditions-custom',
            style={'display': 'none'}
        ),

        html.Div([
            html.Label('Autocall Non Period (Months)'),
            dcc.Input(
                type='number',
                placeholder="12",
                id='input--autocall-non-period'
            ),
        ],
            id='autocall-non-period',
            style={'display': 'none'}
        ),
    ]),

    # ===============================================================
    # SECTION FOR COUPON
    html.Section([
        html.Label('Coupon Section'),

        html.Div([
            html.Label('Coupon Type'),
            dcc.Dropdown(
                options=[
                    {'label': 'Not Applicable', 'value': 'not'},
                    {'label': 'Guaranteed', 'value': 'guaranteed'},
                    {'label': 'Contingent', 'value': 'contingent'},
                    {'label': 'Contingent Memory', 'value': 'memory'},
                ],
                placeholder='Select a coupon type',
                id='input--coupon-type'
            ),
        ]),

        html.Div([
            html.Label('Coupon Conditions'),
            dcc.Dropdown(
                options=[
                    {'label': 'Fixed (Stable for observation Dates)', 'value': 'fixed'},
                    {'label': 'Custom (Different for Dates)', 'value': 'custom'},
                ],
                placeholder='Select coupon conditions',
                id='input--coupon-conditions'
            ),
        ],
            id='coupon-conditions--guaranteed--contingent--memory',
            style={'display': 'none'}
        ),

        html.Div([
            html.Label('Coupon Paid At'),
            dcc.Dropdown(
                options=[
                    {'label': 'Maturity', 'value': 'maturity'},
                    {'label': 'Coupon Date (Immediately)', 'value': 'immediately'}
                ],
                placeholder='Select when coupon paid at',
                id='input--coupon-paid-at'
            ),
        ],
            id='coupon-paid-at--guaranteed--contingent--memory',
            style={'display': 'none'}
        ),

        html.Div([
            html.Label('Coupon Level (%)'),
            dcc.Input(
                type='number',
                placeholder="80",
                id='input--coupon-level--fixed'
            ),
        ],
            id='coupon-level--contingent--memory--conditions-fixed',
            style={'display': 'none'}
        ),

        html.Div([
            html.Label('Coupon Payment'),
            dcc.Input(
                type='number',
                placeholder="5",
                id='input--coupon-payment--fixed'
            ),
            dcc.RadioItems(
                options=['in money', '% of Principal', '% of Principal per annum'],
                inline=True,
                id='input--coupon-payment-type--fixed'
            )
        ],
            id='coupon-payment--guaranteed--contingent--memory--conditions-fixed',
            style={'display': 'none'}
        ),

        html.Div([
            html.Label('Coupon Levels (%)'),
            dcc.Input(
                type='text',
                placeholder="80, 77.5, 75, ...",
                id='input--coupon-level--custom'
            ),
        ],
            id='coupon-level--contingent--memory--conditions-custom',
            style={'display': 'none'}
        ),

        html.Div([
            html.Label('Coupon Payments'),
            dcc.Input(
                type='text',
                placeholder="8, 7.75, 7.5, ...",
                id='input--coupon-payment--custom'
            ),
            dcc.RadioItems(
                options=['in money', '% of Principal', '% of Principal per annum'],
                inline=True,
                id='input--coupon-payment-type--custom'
            )
        ],
            id='coupon-payment--guaranteed--contingent--memory--conditions-custom',
            style={'display': 'none'}
        ),

        html.Div([
            html.Label('Coupon Non Period (Months)'),
            dcc.Input(
                type='number',
                placeholder="1",
                id='input--coupon-non-period'
            ),
        ],
            id='coupon-non-period--guaranteed--contingent--memory',
            style={'display': 'none'}
        ),
    ]),

    # ===============================================================
    # SECTION FOR PROTECTION
    html.Section([
        html.Label('Protection Section'),

        html.Div([
            html.Label('Protection Type'),
            dcc.Dropdown(
                options=[
                    {'label': 'Not Applicable', 'value': 'not'},
                    {'label': 'Guaranteed', 'value': 'full'},
                    {'label': 'Barrier', 'value': 'one'},
                    {'label': 'Barrier + Elite', 'value': 'two'},
                ],
                placeholder='Select a type',
                id='input--protection-type'
            ),
        ]),

        html.Div([
            html.Label('Barrier Type'),
            dcc.Dropdown(
                options=['Soft', 'Hard'],
                placeholder='Select a type',
                id='input--barrier-type',
                value='hard'
            ),
        ],
            id='protection-type--barrier--elite',
            style={'display': 'None'}
        ),

        html.Div([
            html.Label('Barrier Level (%)'),
            dcc.Input(
                type='number',
                placeholder="60",
                id='input--barrier-level'
            ),
        ],
            id='barrier-level--barrier--elite',
            style={'display': 'None'}
        ),

        html.Div([
            html.Label('Elite Barrier Level (%)'),
            dcc.Input(
                type='number',
                placeholder="140",
                id='input--elite-level'
            ),
        ],
            id='elite-level--elite',
            style={'display': 'None'}
        ),

        html.Div([
            html.Label('Elite Barrier Payment'),
            dcc.Input(
                type='number',
                placeholder="140",
                id='input--elite-payment'
            ),
            dcc.RadioItems(
                options=['in money', '% of Principal'],
                inline=True,
                id='input--elite-payment-type'
            )
        ],
            id='elite-payment--elite',
            style={'display': 'None'}
        ),
    ]),

    # ===============================================================
    # SECTION FOR SOLVER SETTINGS
    html.Section([
        html.Label('Solver Settings'),

        html.Div([
            html.Label('Solve For'),
            dcc.Dropdown(
                options=[
                    {'label': 'Theoretical Price', 'value': 'price'},
                    {'label': 'Annualized Yield', 'value': 'yield'},
                ],
                placeholder='Select a problem',
                id='input--solve-for'
            ),
            html.Label("Annualized Yield is available only with coupon payments"),
        ]),

        html.Div([
            html.Label('Selling Commission (%)'),
            dcc.Input(
                type='number',
                placeholder="0.33",
                id='input--commission',
            ),
            dcc.RadioItems(
                options=['at once', 'per annum'],
                inline=True,
                id='input--commission-type'
            )
        ],
            id='commission--solve-yield',
            style={'display': 'none'}
        ),

        html.Div([
            html.Label('Pricing Method'),
            dcc.Dropdown(
                options=[
                    {'label': 'Constant Volatility (GBM)', 'value': 'constant'},
                    {'label': 'Stochastic Volatility (Heston)', 'value': 'stochastic'},
                ],
                placeholder='Select a method',
                id='input--pricing-method'
            ),
            html.Label("Stochastic Volatility is available only for one underlying"),
        ]),

        html.Div([
            html.Button('Calculate', id='button--calculate', n_clicks=0),
            html.Button('Clear', id='button--clear', n_clicks=0)
        ])
    ]),

    # ===============================================================
    # SECTION FOR RESULTS
    html.Section([
        html.Div(id='result-ind')
    ]),

    html.Div([html.P("."), html.P(".")])
])


# Choose calculator mode
@app.callback(
    [
        Output(component_id='asset--mode-ticker', component_property='style'),
        Output(component_id='currency--mode-ticker', component_property='style'),
        Output(component_id='asset--mode-custom', component_property='style'),
        Output(component_id='currency--mode-custom', component_property='style')
    ],
    [
        Input(component_id='input--mode', component_property='value')
    ]
)
def show_hide_asset_inputs(app_mode):
    if app_mode == 'Custom':
        return (
            {'display': 'none'}, {'display': 'none'},
            {'display': 'block'}, {'display': 'block'}
        )

    elif app_mode == 'Ticker':
        return (
            {'display': 'block'}, {'display': 'block'},
            {'display': 'none'}, {'display': 'none'}
        )


# Choose autocall conditions
@app.callback(
    [
        Output(component_id='autocall-level--conditions-fixed', component_property='style'),
        Output(component_id='autocall-level--conditions-custom', component_property='style'),
        Output(component_id='autocall-payment--conditions-custom', component_property='style'),
        Output(component_id='autocall-non-period', component_property='style')
    ],
    [
        Input(component_id='input--autocall-conditions', component_property='value')
    ]
)
def show_hide_autocall_inputs(condition):
    if condition == 'custom':
        return (
            {'display': 'none'}, {'display': 'block'},
            {'display': 'block'}, {'display': 'block'}
        )

    elif condition == 'fixed':
        return (
            {'display': 'block'}, {'display': 'none'},
            {'display': 'none'}, {'display': 'block'}
        )

    return (
        {'display': 'none'}, {'display': 'none'},
        {'display': 'none'}, {'display': 'none'}
    )


@app.callback(
        Output(component_id='commission--solve-yield', component_property='style')
    ,
    [
        Input(component_id='input--solve-for', component_property='value'),
        Input(component_id='input--coupon-type', component_property='value')
    ]
)
def show_hide_commission_inputs(solve_for, coupon_type):
    if coupon_type in ('guaranteed', 'contingent', 'memory') and solve_for == 'yield':
        return {'display': 'block'}

    return {'display': 'none'}


# Choose coupon type
@app.callback(
    [
        Output(component_id='coupon-paid-at--guaranteed--contingent--memory', component_property='style'),
        Output(component_id='coupon-conditions--guaranteed--contingent--memory', component_property='style'),
        Output(component_id='coupon-non-period--guaranteed--contingent--memory', component_property='style')
    ],
    [
        Input(component_id='input--coupon-type', component_property='value'),
        Input(component_id='input--coupon-conditions', component_property='value')
    ]
)
def show_hide_coupon_inputs(coupon_type, coupon_conditions):
    if coupon_type in ('guaranteed', 'contingent', 'memory'):
        return (
            {'display': 'block'}, {'display': 'block'},
            {'display': 'block'}
        )

    return (
        {'display': 'none'}, {'display': 'none'},
        {'display': 'none'}

    )


# Choose coupon conditions
@app.callback(
    [
        Output(component_id='coupon-level--contingent--memory--conditions-fixed', component_property='style'),
        Output(component_id='coupon-payment--guaranteed--contingent--memory--conditions-fixed',
               component_property='style'),
        Output(component_id='coupon-level--contingent--memory--conditions-custom', component_property='style'),
        Output(component_id='coupon-payment--guaranteed--contingent--memory--conditions-custom',
               component_property='style')
    ],
    [
        Input(component_id='input--coupon-type', component_property='value'),
        Input(component_id='input--coupon-conditions', component_property='value')
    ]
)
def show_hide_coupon1_inputs(coupon_type, coupon_conditions):
    if coupon_type == 'guaranteed' and coupon_conditions == 'fixed':
        return (
            {'display': 'none'}, {'display': 'block'},
            {'display': 'none'}, {'display': 'none'}

        )

    elif coupon_type == 'guaranteed' and coupon_conditions == 'custom':
        return (
            {'display': 'none'}, {'display': 'none'},
            {'display': 'none'}, {'display': 'block'}

        )

    elif coupon_type in ('contingent', 'memory') and coupon_conditions == 'fixed':
        return (
            {'display': 'block'}, {'display': 'block'},
            {'display': 'none'}, {'display': 'none'}

        )

    elif coupon_type in ('contingent', 'memory') and coupon_conditions == 'custom':
        return (
            {'display': 'none'}, {'display': 'none'},
            {'display': 'block'}, {'display': 'block'}

        )

    return (
        {'display': 'none'}, {'display': 'none'},
        {'display': 'none'}, {'display': 'none'}

    )


# Choose protection conditions
@app.callback(
    [
        Output(component_id='protection-type--barrier--elite', component_property='style'),
        Output(component_id='barrier-level--barrier--elite', component_property='style'),
        Output(component_id='elite-level--elite', component_property='style'),
        Output(component_id='elite-payment--elite', component_property='style')
    ],
    [
        Input(component_id='input--protection-type', component_property='value')
    ]
)
def show_hide_protection_inputs(protection_type):
    if protection_type in ('not', 'full'):
        return (
            {'display': 'none'}, {'display': 'none'},
            {'display': 'none'}, {'display': 'none'},
        )

    elif protection_type == 'one':
        return (
            {'display': 'block'}, {'display': 'block'},
            {'display': 'none'}, {'display': 'none'},
        )

    elif protection_type == 'two':
        return (
            {'display': 'block'}, {'display': 'block'},
            {'display': 'block'}, {'display': 'block'},
        )
    return (
        {'display': 'none'}, {'display': 'none'},
        {'display': 'none'}, {'display': 'none'},
    )


@app.callback(
    [
        Output(component_id='result-ind', component_property='children'),
        Output(component_id='button--calculate', component_property='n_clicks')
    ],
    [  # all inputs in calculator
        Input(component_id='input--mode', component_property='value'),

        Input(component_id='input--underlying', component_property='value'),
        Input(component_id='input--initial-fixing-date', component_property='date'),
        Input(component_id='input--spot-prices', component_property='value'),
        Input(component_id='input--dividend-yields', component_property='value'),
        Input(component_id='input--volatilities', component_property='value'),
        Input(component_id='input--correlations', component_property='value'),

        Input(component_id='input--term', component_property='value'),
        Input(component_id='input--weighting', component_property='value'),
        Input(component_id='input--observation-freq', component_property='value'),
        Input(component_id='input--principal', component_property='value'),

        Input(component_id='input--currency', component_property='value'),
        Input(component_id='input--risk-free-rate-tenors', component_property='value'),
        Input(component_id='input--risk-free-rate-values', component_property='value'),

        Input(component_id='input--autocall-conditions', component_property='value'),
        Input(component_id='input--autocall-level--fixed', component_property='value'),
        Input(component_id='input--autocall-level--custom', component_property='value'),
        Input(component_id='input--autocall-payment--custom', component_property='value'),
        Input(component_id='input--autocall-payment-type--custom', component_property='value'),
        Input(component_id='input--autocall-non-period', component_property='value'),

        Input(component_id='input--coupon-type', component_property='value'),
        Input(component_id='input--coupon-conditions', component_property='value'),
        Input(component_id='input--coupon-paid-at', component_property='value'),
        Input(component_id='input--coupon-level--fixed', component_property='value'),
        Input(component_id='input--coupon-payment--fixed', component_property='value'),
        Input(component_id='input--coupon-payment-type--fixed', component_property='value'),
        Input(component_id='input--coupon-level--custom', component_property='value'),
        Input(component_id='input--coupon-payment--custom', component_property='value'),
        Input(component_id='input--coupon-payment-type--custom', component_property='value'),
        Input(component_id='input--coupon-non-period', component_property='value'),

        Input(component_id='input--protection-type', component_property='value'),
        Input(component_id='input--barrier-type', component_property='value'),
        Input(component_id='input--barrier-level', component_property='value'),
        Input(component_id='input--elite-level', component_property='value'),
        Input(component_id='input--elite-payment', component_property='value'),
        Input(component_id='input--elite-payment-type', component_property='value'),

        Input(component_id='input--solve-for', component_property='value'),
        Input(component_id='input--commission', component_property='value'),
        Input(component_id='input--commission-type', component_property='value'),
        Input(component_id='input--pricing-method', component_property='value'),
        Input(component_id='button--calculate', component_property='n_clicks')
    ]
)
def calculate_button(mode,
                     underlying, initial_fixing_date,
                     spot_prices, dividend_yields, volatilities, correlations,
                     term, weighting, observation_freq, principal,
                     currency, risk_free_rate_tenors, risk_free_rate_values,

                     autocall_conditions, autocall_level__fixed, autocall_level__custom,
                     autocall_payment__custom, autocall_payment_type__custom, autocall_non_period,

                     coupon_type, coupon_conditions, coupon_paid_at,
                     coupon_level__fixed, coupon_payment__fixed, coupon_payment_type__fixed,
                     coupon_level__custom, coupon_payment__custom, coupon_payment_type__custom,
                     coupon_non_period,

                     protection_type, barrier_type, barrier_level, elite_level,
                     elite_payment, elite_payment_type,

                     solve_for, commission, commission_type, pricing_method, n_clicks):

    if "button--calculate" == ctx.triggered_id:
        print(underlying, initial_fixing_date, currency)

        params_for_calc, bondfloor = validate_inputs(locals())

        if params_for_calc[-1] == 'price':
            result = get_present_value(*params_for_calc[:-2])

        if params_for_calc[-1] == 'yield':
            result = get_coupon_annual_yield(*params_for_calc[:-1])

        params_name = ['principal', 'discount_rate', 'performance', 'observation_dates',
                       'autocall_level', 'autocall_payment', 'autocall_non_period',
                       'protection_type', 'barrier_type', 'barrier_level', 'elite_level', 'elite_payment',
                       'coupon_type', 'coupon_paid_at', 'coupon_non_period', 'coupon_level', 'coupon_payment',
                       'implied_PV', 'solve_for', 'bondfloor', 'result']

        result_text = []
        for name, param in zip(params_name[::-1], [result, bondfloor] +params_for_calc[::-1]):
            result_text.append(html.P(f"{name}: {param}"))

        return result_text, 0

    else:
        return [], 0


if __name__ == '__main__':
    app.run_server(debug=True)
